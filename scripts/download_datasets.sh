#!/usr/bin/env bash
# Fetch the public benchmark datasets (all hosted on GitHub) into ${1:-data}/
# Licenses: see each repository. Magnetic-tile: research use; BSData: CC BY 4.0;
# DeepCrack / CrackForest: research use. Do not redistribute.
set -euo pipefail
DEST=${1:-data}
mkdir -p "$DEST"
cd "$DEST"
clone() { [ -d "$2" ] || git clone --depth 1 "https://github.com/$1.git" "$2"; }
clone abin24/Magnetic-tile-defect-datasets. Magnetic-tile-defect-datasets.
clone 2Obe/BSData BSData
clone cuilimeng/CrackForest-dataset CrackForest-dataset
if [ ! -d DeepCrack ]; then
  clone yhlleo/DeepCrack DeepCrack-repo
  mkdir -p DeepCrack
  python3 -I -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      DeepCrack-repo/dataset/DeepCrack.zip DeepCrack
fi
echo "datasets in $(pwd)"
