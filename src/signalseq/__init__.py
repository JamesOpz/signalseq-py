"""signalseq -- YAML-driven ADT quantification for SIGNAL-seq libraries.

Every chemistry-specific value (read offsets and lengths, barcode whitelists,
mismatch tolerances, filter thresholds) comes from a YAML config; nothing is
hardcoded here. See configs/parse_v3.example.yaml for the Parse Evercode WT v3
chemistry, and the README for adapting it to another chemistry.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("signalseq")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
