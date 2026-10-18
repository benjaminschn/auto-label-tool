"""End-to-end auto-labeling: cross-fitted ensemble -> calibration -> thresholds with recall guarantee
-> fold-consistent self-training -> self-assessment and review ranking.
"""
from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from . import conformal as C
from . import metrics as M
from .calibration import IsotonicCalibrator, expected_calibration_error
from .data import Sample, kfold_indices
from .infer import predict_image
from .postprocess import hysteresis, select_thresholds
from .quality import CandidateModel, image_quality, review_risk
from .train import TrainConfig, TrainData, train_member


@dataclass
class AutoLabelConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    k_folds: int = 4
    rounds: int = 2                      # 1 = no self-training
    st_iters: Optional[int] = None       # iterations of self-training rounds (default: train.iters)
    p_pseudo: float = 0.4                # share of pseudo-labeled samples during self-training
    use_anomaly: bool = True
    anomaly_backbone: str = "wide_resnet50_2"
    anomaly_patches_per_image: int = 64
    bg_clean_frac: float = 0.7           # use the most normal-looking share of U as synthesis backgrounds
    bg_ignore_quantile: float = 0.98     # round 0: ignore the most anomalous pixels of backgrounds (may be
                                         # unlabeled real defects); later rounds use teacher pseudo-labels
    tta: str = "hflip"
    tol: int = 2
    merge_radius: int = 2
    threshold_objective: str = "tol_f1"
    alpha: float = 0.1                   # tolerated share of missed defects for the recall guarantee
    delta: float = 0.1                   # confidence level of the PAC guarantee
    ignore_ratio: float = 0.5            # pseudo-labels: ignore pixels with p in [ratio*t_ext, label)
    ignore_ring: int = 2                 # pseudo-labels: ignore a ring around predicted defects
    teacher: str = "fold"                # "fold" (honest, diverse) or "ensemble"
    keep_member_probs: bool = True
    seed: int = 0

    def to_dict(self):
        d = dataclasses.asdict(self)
        return d


def _log_default(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class AutoLabeler:
    def __init__(self, cfg: AutoLabelConfig, log: Callable[[str], None] = _log_default,
                 workdir: Optional[str] = None):
        self.cfg, self.log, self.workdir = cfg, log, workdir
        if workdir:
            os.makedirs(workdir, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _pseudo_labels(self, p: np.ndarray, th: Dict[str, float]):
        pos = hysteresis(p, th["t_seed"], th["t_ext"], int(th["min_area"]))
        ign = (p >= self.cfg.ignore_ratio * th["t_ext"]) & ~pos
        if self.cfg.ignore_ring > 0 and pos.any():
            ign |= M.dilate(pos, self.cfg.ignore_ring) & ~pos
        return pos, (~ign).astype(np.float32)

    # ------------------------------------------------------------------ main
    def run(self, labeled: Sequence[Sample], unlabeled: Sequence[Sample],
            extra: Optional[Sequence[Sample]] = None) -> Dict:
        cfg, log = self.cfg, self.log
        t_start = time.time()
        L_img = [s.image for s in labeled]
        L_msk = [s.mask.astype(bool) for s in labeled]
        U_img = [s.image for s in unlabeled]
        X_img = [s.image for s in extra] if extra else []
        res: Dict = {"config": cfg.to_dict(), "labeled_ids": [s.id for s in labeled],
                     "unlabeled_ids": [s.id for s in unlabeled],
                     "extra_ids": [s.id for s in extra] if extra else [], "rounds": []}

        # 1) normality model from the unlabeled pool
        anomaly_U = anomaly_L = None
        bg_idx = list(range(len(U_img)))
        bg_labels0 = None
        if cfg.use_anomaly:
            from .anomaly import NormalityModel
            t0 = time.time()
            nm = NormalityModel(cfg.anomaly_backbone, patches_per_image=cfg.anomaly_patches_per_image,
                                seed=cfg.seed, threads=cfg.train.threads).fit(U_img, log=log)
            anomaly_U = nm.score_many(U_img, list(range(len(U_img))), log=log)
            anomaly_L = nm.score_many(L_img)
            del nm
            img_score = np.array([float(a.max()) for a in anomaly_U])
            order = np.argsort(img_score)
            bg_idx = sorted(order[: max(1, int(round(cfg.bg_clean_frac * len(order))))].tolist())
            if cfg.bg_ignore_quantile < 1:
                sub = np.concatenate([a.astype(np.float32).ravel()[::11] for a in anomaly_U])
                thr = float(np.quantile(sub, cfg.bg_ignore_quantile))
                bg_labels0 = [(np.zeros(U_img[i].shape[:2], bool),
                               (anomaly_U[i].astype(np.float32) < thr).astype(np.float32)) for i in bg_idx]
            res["anomaly_image_score_U"] = img_score
            res["anomaly_image_score_L"] = np.array([float(a.max()) for a in anomaly_L])
            log(f"normality model done in {time.time() - t0:.0f}s; {len(bg_idx)} clean backgrounds")
        res["background_idx"] = bg_idx

        folds = kfold_indices(list(range(len(labeled))), list(labeled), cfg.k_folds, cfg.seed)
        folds = [f for f in folds if f]
        teacher_members: Optional[List[np.ndarray]] = None
        teacher_th = None

        for r in range(cfg.rounds):
            t0 = time.time()
            log(f"=== round {r} ({'supervised + synthetic' if r == 0 else 'self-training'}) ===")
            oof: List[Optional[np.ndarray]] = [None] * len(labeled)
            member_U = [[] for _ in U_img]
            member_X = [[] for _ in X_img]
            for k, fold in enumerate(folds):
                fold_set = set(fold)
                train_L = [(L_img[j], L_msk[j]) for j in range(len(labeled)) if j not in fold_set]
                if not train_L:   # leave-one-out with a single labeled image: train on it anyway
                    train_L = [(L_img[j], L_msk[j]) for j in range(len(labeled))]
                data = TrainData(labeled=train_L, backgrounds=[U_img[i] for i in bg_idx],
                                 background_labels=bg_labels0)
                tcfg = dataclasses.replace(cfg.train, seed=cfg.seed * 1000 + r * 100 + k)
                if r > 0:
                    t_probs = [m[k] if cfg.teacher == "fold" else m.mean(0) for m in teacher_members]
                    pl = [self._pseudo_labels(p.astype(np.float32), teacher_th) for p in t_probs]
                    data.pseudo = [(U_img[i], pl[i][0], pl[i][1]) for i in range(len(U_img))]
                    data.background_labels = [pl[i] for i in bg_idx]
                    tcfg = dataclasses.replace(tcfg, p_pseudo=cfg.p_pseudo,
                                               iters=cfg.st_iters or cfg.train.iters)
                log(f"member {k}: {len(train_L)} labeled imgs, fold {len(fold)}")
                model = train_member(data, tcfg, log=log)
                for j in fold:
                    oof[j] = predict_image(model, L_img[j], cfg.tta)
                for i, img in enumerate(U_img):
                    member_U[i].append(predict_image(model, img, cfg.tta).astype(np.float16))
                for i, img in enumerate(X_img):
                    member_X[i].append(predict_image(model, img, cfg.tta).astype(np.float16))
                if self.workdir:
                    torch.save(model.state_dict(), os.path.join(self.workdir, f"member_r{r}_k{k}.pt"))
                del model
            member_U = [np.stack(m) for m in member_U]
            mean_U = [m.astype(np.float32).mean(0) for m in member_U]
            mean_X = [np.stack(m).astype(np.float32).mean(0) for m in member_X]

            # 2) calibration + thresholds on out-of-fold predictions
            cal = IsotonicCalibrator().fit(oof, L_msk)
            th, oof_m = select_thresholds(oof, L_msk, cfg.tol, cfg.merge_radius,
                                          objective=cfg.threshold_objective)
            dscores = C.defect_scores(oof, L_msk, cfg.tol, cfg.merge_radius)
            t_conf = C.conformal_threshold(dscores, cfg.alpha)
            t_pac = C.pac_threshold(dscores, cfg.alpha, cfg.delta)
            cert = C.certified_recall(dscores, th["t_seed"], cfg.delta)
            log(f"OOF on L: tol_f1={oof_m['tol_f1']:.3f} defect_recall={oof_m['defect_recall']:.3f} "
                f"thresholds={th}  conformal t={t_conf:.3f}  pac t={t_pac:.3f}  "
                f"({len(dscores)} labeled defects; recall guarantee at t_seed: marginal {cert[0]:.2f}, "
                f"PAC {cert[1]:.2f})")
            masks_U = [hysteresis(p, th["t_seed"], th["t_ext"], int(th["min_area"])) for p in mean_U]
            t_safe = min(th["t_seed"], t_conf) if t_conf > 0 else th["t_seed"]
            safe_U = [hysteresis(p, t_safe, min(th["t_ext"], t_safe), int(th["min_area"])) for p in mean_U]
            rr = {
                "round": r, "thresholds": th, "oof_metrics": oof_m,
                "oof_ece": expected_calibration_error(oof, L_msk),
                "defect_scores_oof": dscores, "t_conformal": t_conf, "t_pac": t_pac,
                "certified_recall_at_t_seed": cert, "t_safe": t_safe,
                "prob_U": [p.astype(np.float16) for p in mean_U], "mask_U": masks_U, "mask_safe_U": safe_U,
                "prob_X": [p.astype(np.float16) for p in mean_X],
                "oof_prob": [p.astype(np.float16) for p in oof],
                "seconds": time.time() - t0,
            }
            if cfg.keep_member_probs:
                rr["member_U"] = member_U
            res["rounds"].append(rr)
            teacher_members, teacher_th = member_U, th
            log(f"round {r} finished in {time.time() - t0:.0f}s")

        # 3) self-assessment of the final labels
        last = res["rounds"][-1]
        th, oof = last["thresholds"], [p.astype(np.float32) for p in last["oof_prob"]]
        cal = IsotonicCalibrator().fit(oof, L_msk)
        cand = CandidateModel(t_low=max(0.02, 0.25 * th["t_ext"]), tol=cfg.tol,
                              merge_radius=cfg.merge_radius).fit(oof, L_msk)
        feats = []
        for i in range(len(U_img)):
            p = last["prob_U"][i].astype(np.float32)
            feats.append(image_quality(p, cal(p), last["mask_U"][i], cand, teacher_members[i],
                                       None if anomaly_U is None else anomaly_U[i], th["t_ext"]))
        oof_feats = []
        for j in range(len(labeled)):
            oof_feats.append(image_quality(oof[j], cal(oof[j]),
                                           hysteresis(oof[j], th["t_seed"], th["t_ext"], int(th["min_area"])),
                                           cand, None, None if anomaly_L is None else anomaly_L[j], th["t_ext"]))
        res["features_U"] = feats
        res["features_oof"] = oof_feats
        res["risk_U"] = review_risk(feats)
        res["candidate_no_cand_rate"] = cand.no_candidate_rate
        if anomaly_U is not None:
            res["anomaly_U"] = anomaly_U
            res["anomaly_L"] = anomaly_L
        res["seconds"] = time.time() - t_start
        log(f"auto-labeling done in {res['seconds'] / 60:.1f} min")
        return res
