#!/usr/bin/env bash
set -euo pipefail

echo "Upgrade pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Install numpy first (binary wheel)..."
python -m pip install --prefer-binary --upgrade --force-reinstall numpy==1.25.2

echo "Install other requirements (prefer binary wheels)..."
python -m pip install --prefer-binary -r requirements.txt

echo "Sanity check versions..."
python - <<'PY'
import sys
import numpy
print("python:", sys.version.splitlines()[0])
print("numpy:", numpy.__version__, numpy.__file__)
try:
    import tensorflow as tf
    print("tf:", tf.__version__)
except Exception as e:
    print("tf import failed:", e)
try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except Exception as e:
    print("sklearn import failed:", e)
PY

echo "build.sh finished."
