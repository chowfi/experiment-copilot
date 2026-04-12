#!/usr/bin/env bash
# Download and unzip the Kaggle customer-churn dataset into ./data (matches customer_churn_random_forest.py).
# Requires a Kaggle API token in ~/.kaggle/kaggle.json (same as the kaggle CLI).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ZIP="${DATA_DIR}/customer-churn-dataset.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/muhammadshahidazeem/customer-churn-dataset"

mkdir -p "${DATA_DIR}"

if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  KAGGLE_USER="$(python3 -c "import json, pathlib; p=pathlib.Path.home()/'.kaggle/kaggle.json'; print(json.loads(p.read_text())['username'])")"
  KAGGLE_KEY="$(python3 -c "import json, pathlib; p=pathlib.Path.home()/'.kaggle/kaggle.json'; print(json.loads(p.read_text())['key'])")"
  curl -L -u "${KAGGLE_USER}:${KAGGLE_KEY}" -o "${ZIP}" "${URL}"
else
  echo "Note: ~/.kaggle/kaggle.json not found. If the download fails, create API credentials at https://www.kaggle.com/settings" >&2
  curl -L -o "${ZIP}" "${URL}"
fi

unzip -o "${ZIP}" -d "${DATA_DIR}"
rm -f "${ZIP}"
echo "Done. CSVs are under: ${DATA_DIR}"
