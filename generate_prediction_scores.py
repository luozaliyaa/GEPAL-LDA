"""Export fold-level prediction logs to a machine-readable CSV file.

The script does not alter the source data, weights, or logs. It parses the
existing metrics/fold_*/predict_log_fold_*.txt files and writes a combined
CSV plus a small provenance manifest under metrics/10fold/.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from experiment_paths import DATA_ROOT

ROOT = Path(__file__).resolve().parent
DATA_DIR = DATA_ROOT
METRICS_DIR = ROOT / "metrics"
OUTPUT_DIR = METRICS_DIR / "10fold"

LOG_PATTERN = re.compile(
    r"lnc_id=(?P<lnc_id>\d+) \| drug_id=(?P<drug_node_id>\d+) "
    r"\| response=(?P<predicted_label>.*?) \| answer=(?P<true_label>.*?) "
    r"\|\s+true_score=(?P<probability_true>[0-9.]+) "
    r"\| false_score=(?P<probability_false>[0-9.]+)",
    flags=re.DOTALL,
)


def read_index_map(path: Path, name_column: str) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["Index"]): row[name_column].strip()
            for row in csv.DictReader(handle)
        }


def main() -> None:
    lnc_map = read_index_map(DATA_DIR / "lncRNA_index.csv", "lncRNA")
    drug_map = read_index_map(DATA_DIR / "drug_index.csv", "Drug")
    drug_offset = len(lnc_map)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    fold_manifest: list[dict[str, object]] = []
    for fold in range(1, 11):
        log_path = METRICS_DIR / f"fold_{fold}" / f"predict_log_fold_{fold}.txt"
        if not log_path.exists():
            raise FileNotFoundError(f"Prediction log not found: {log_path}")

        parsed = 0
        log_text = log_path.read_text(encoding="utf-8", errors="strict")
        for parsed, match in enumerate(LOG_PATTERN.finditer(log_text), start=1):
            lnc_id = int(match["lnc_id"])
            drug_node_id = int(match["drug_node_id"])
            drug_id = drug_node_id - drug_offset
            if lnc_id not in lnc_map or drug_id not in drug_map:
                raise ValueError(
                    f"Unmapped entity at fold {fold}, prediction record {parsed}: "
                    f"lnc_id={lnc_id}, drug_node_id={drug_node_id}"
                )
            rows.append(
                {
                    "fold": fold,
                    "record_index": parsed,
                    "lnc_id": lnc_id,
                    "lncRNA": lnc_map[lnc_id],
                    "drug_node_id": drug_node_id,
                    "drug_id": drug_id,
                    "drug": drug_map[drug_id],
                    "true_label": match["true_label"].strip(),
                    "predicted_label": match["predicted_label"].strip(),
                    "probability_true": match["probability_true"],
                    "probability_false": match["probability_false"],
                    "source_log": log_path.relative_to(ROOT).as_posix(),
                }
            )
        fold_manifest.append(
            {
                "fold": fold,
                "source_log": log_path.relative_to(ROOT).as_posix(),
                "parsed_prediction_records": parsed,
            }
        )

    output_csv = OUTPUT_DIR / "prediction_scores.csv"
    fieldnames = [
        "fold",
        "record_index",
        "lnc_id",
        "lncRNA",
        "drug_node_id",
        "drug_id",
        "drug",
        "true_label",
        "predicted_label",
        "probability_true",
        "probability_false",
        "source_log",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "output_file": output_csv.relative_to(ROOT).as_posix(),
        "total_prediction_records": len(rows),
        "probability_precision": "Parsed from source logs, which record four decimal places.",
        "folds": fold_manifest,
    }
    manifest_path = OUTPUT_DIR / "prediction_scores_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote provenance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
