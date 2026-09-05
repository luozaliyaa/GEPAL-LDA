"""Verify the archived cross-validation partitions used by the manuscript.

The former utility regenerated ten folds by shuffling the input data. That
would produce a new partition and could overwrite the archived experiment
splits, so it must not be used to reproduce the reported results.
"""

from __future__ import annotations

import json

from experiment_paths import FOLD_ROOT


def verify_archived_splits() -> dict[int, dict[str, int]]:
    """Return and print the sample counts of the immutable archived folds."""
    counts: dict[int, dict[str, int]] = {}
    for fold in range(1, 11):
        fold_dir = FOLD_ROOT / f"fold_{fold}"
        counts[fold] = {}
        for split in ("train", "val", "test"):
            path = fold_dir / f"lnc_drug_{split}.json"
            with path.open("r", encoding="utf-8") as handle:
                counts[fold][split] = len(json.load(handle))
        print(
            f"[Fold {fold}] train={counts[fold]['train']} "
            f"val={counts[fold]['val']} test={counts[fold]['test']}"
        )
    print(f"Archived folds verified: {FOLD_ROOT}")
    return counts


if __name__ == "__main__":
    verify_archived_splits()
