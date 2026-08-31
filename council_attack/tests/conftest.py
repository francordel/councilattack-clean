"""Make the in-tree package importable without installation. The tests exercise
the pure pipeline collaborators directly and require no network."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
