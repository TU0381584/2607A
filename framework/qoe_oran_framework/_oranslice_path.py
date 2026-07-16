"""Makes drl_slicing/oranslice_drl importable regardless of caller CWD.

qoe_oran_framework lives at the repo root, alongside drl_slicing/ (not
inside it), so `import oranslice_drl` only works once drl_slicing/ is on
sys.path. Call ensure_oranslice_drl_importable() before any such import.
"""

import sys
from pathlib import Path

_DRL_SLICING_DIR = Path(__file__).resolve().parent.parent / "drl_slicing"


def ensure_oranslice_drl_importable() -> None:
    path_str = str(_DRL_SLICING_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
