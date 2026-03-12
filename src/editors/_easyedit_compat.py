"""
Compatibility layer to import EasyEdit apply functions from src_orig/.

The EasyEdit root __init__.py triggers cascading imports of all methods,
which fails due to missing dependencies (e.g., 'higher' for MEND).
This module registers the EasyEdit package hierarchy as namespace packages
(with empty __init__) so that individual apply functions can be imported
without triggering the cascade.
"""
import sys
import types
from pathlib import Path

_src_orig = Path(__file__).resolve().parents[2] / "src_orig"

_NAMESPACE_PACKAGES = [
    "easyeditor",
    "easyeditor.util",
    "easyeditor.models",
    "easyeditor.models.ft",
    "easyeditor.models.rome",
    "easyeditor.models.memit",
    "easyeditor.models.alphaedit",
]

for _pkg in _NAMESPACE_PACKAGES:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [str(_src_orig / _pkg.replace(".", "/"))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod
