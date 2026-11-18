import sys
from pathlib import Path

# Make the pipeline's shared modules (scripts/funcs.py, scripts/common.py) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
