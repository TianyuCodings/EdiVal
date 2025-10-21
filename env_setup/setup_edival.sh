#!/usr/bin/env bash
set -euo pipefail

# Portable setup script that bootstraps the `edival` conda environment from env.yaml,
# installs Grounding DINO locally, and fetches the default weights.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.yaml"
ENV_NAME="edival"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found on PATH. Install Miniconda or Mambaforge first." >&2
  exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
  echo "Environment file not found at ${ENV_FILE}" >&2
  exit 1
fi

echo ">>> Creating or updating conda environment '${ENV_NAME}' from ${ENV_FILE}"
if conda env list | awk '{print $1}' | grep -Fx "${ENV_NAME}" >/dev/null 2>&1; then
  echo "Environment '${ENV_NAME}' already exists. Updating packages to match env.yaml."
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -f "${ENV_FILE}"
fi

echo ">>> Upgrading pip inside '${ENV_NAME}'"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo ">>> Installing GroundingDINO in editable mode"
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}/GroundingDINO" --config-settings editable_mode=compat

echo ">>> Ensuring Grounding DINO weights are present"
export EDIVAL_REPO_ROOT="${REPO_ROOT}"
conda run -n "${ENV_NAME}" python - <<'PY'
import os
import pathlib
import urllib.request

repo_root = pathlib.Path(os.environ["EDIVAL_REPO_ROOT"])
weights_dir = repo_root / "GroundingDINO" / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)
weights_path = weights_dir / "groundingdino_swint_ogc.pth"
if weights_path.exists():
    print(f"Weight file already exists at {weights_path}")
else:
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    print(f"Downloading {url} -> {weights_path}")
    urllib.request.urlretrieve(url, weights_path)
PY
unset EDIVAL_REPO_ROOT

cat <<'MSG'
------------------------------------------------------------
Environment bootstrap complete.
Activate it with:   conda activate edival
MSG
