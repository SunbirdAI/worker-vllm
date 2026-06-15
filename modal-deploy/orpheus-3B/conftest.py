"""Top-level pytest conftest: makes the orpheus-3B/api package importable
when pytest is run from this directory."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
