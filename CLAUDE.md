# auto-label-tool – Hinweise für Claude Code

- Projektstand, offene Punkte und Experimentplan: `docs/HANDOFF.md` (zuerst lesen).
- Methode und Begründungen: `docs/METHOD.md`. Dokumentation ist auf Deutsch, Code/Kommentare auf Englisch.
- Tests: `python -m pytest -q` (schnell, CPU).
- Benchmarks sind teuer: erst Mini-Smoke-Test (`--iters 20 --k-folds 2 --max-unlabeled 40`), dann volle Läufe.
- Umgebungsvariablen: `AUTOLABEL_DATA`, `AUTOLABEL_CACHE`, `AUTOLABEL_WEIGHTS`, `AUTOLABEL_DEVICE` (cpu|mps|cuda).
- Datensätze, Laufergebnisse (`runs/`), Caches und Gewichte nie committen (siehe `.gitignore`).
- Ergebnisse nur aus echten Läufen berichten; ungetestete Teile als solche kennzeichnen.
