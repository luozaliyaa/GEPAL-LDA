"""Canonical paths for the archived experiment version used in the manuscript.

The ``_rf`` suffix is retained because it denotes the reliable-negative
construction. It is not an alias for the older ``D-lnc_with_features``
dataset: this module points to the preserved 65,432-record experiment version.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "D-lnc_with_features_experiment_v1_1"
FOLD_ROOT = DATA_ROOT / "10fold_rf"
