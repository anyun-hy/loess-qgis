import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "inference_scripts", ROOT / "qgis_plugins"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

