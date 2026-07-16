import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class RFSamplingSummary:
    num_lnc: int
    num_drug: int
    num_positive_pairs: int
    num_unlabeled_pairs: int
    num_selected_negatives: int
    pair_feature_dim: int
    lnc_pca_dim: int
    drug_pca_dim: int
    initial_negative_multiplier: int
    provisional_negative_multiplier: int
    n_estimators: int
    max_depth: int | None
    random_seed: int


def load_feature_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter="\t", dtype=np.float32)


def resolve_pca_dim(requested_dim: int, feature_matrix: np.ndarray) -> int:
    return max(1, min(requested_dim, feature_matrix.shape[0], feature_matrix.shape[1]))


def fit_pca_features(lnc_features: np.ndarray, drug_features: np.ndarray, pca_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    shared_dim = min(resolve_pca_dim(pca_dim, lnc_features), resolve_pca_dim(pca_dim, drug_features))

    lnc_pca = PCA(n_components=shared_dim, random_state=42)
    drug_pca = PCA(n_components=shared_dim, random_state=42)

    lnc_reduced = lnc_pca.fit_transform(lnc_features).astype(np.float32)
    drug_reduced = drug_pca.fit_transform(drug_features).astype(np.float32)
    return lnc_reduced, drug_reduced


def build_pair_features(
    pairs: np.ndarray,
    lnc_embeddings: np.ndarray,
    drug_embeddings: np.ndarray,
) -> np.ndarray:
    lnc_vecs = lnc_embeddings[pairs[:, 0]]
    drug_vecs = drug_embeddings[pairs[:, 1]]
    abs_diff = np.abs(lnc_vecs - drug_vecs)
    elem_prod = lnc_vecs * drug_vecs
    return np.concatenate([lnc_vecs, drug_vecs, abs_diff, elem_prod], axis=1).astype(np.float32)


def train_rf(
    pos_pairs: np.ndarray,
    neg_pairs: np.ndarray,
    lnc_embeddings: np.ndarray,
    drug_embeddings: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    random_seed: int,
) -> RandomForestClassifier:
    x_pos = build_pair_features(pos_pairs, lnc_embeddings, drug_embeddings)
    x_neg = build_pair_features(neg_pairs, lnc_embeddings, drug_embeddings)
    x_train = np.concatenate([x_pos, x_neg], axis=0)
    y_train = np.concatenate(
        [
            np.ones(len(x_pos), dtype=np.int32),
            np.zeros(len(x_neg), dtype=np.int32),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(x_train, y_train)
    return rf


def score_unlabeled_pairs(
    rf: RandomForestClassifier,
    unlabeled_pairs: np.ndarray,
    lnc_embeddings: np.ndarray,
    drug_embeddings: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    scores = np.empty(len(unlabeled_pairs), dtype=np.float32)
    for start in range(0, len(unlabeled_pairs), chunk_size):
        end = min(start + chunk_size, len(unlabeled_pairs))
        chunk_pairs = unlabeled_pairs[start:end]
        chunk_features = build_pair_features(chunk_pairs, lnc_embeddings, drug_embeddings)
        chunk_scores = rf.predict_proba(chunk_features)[:, 1]
        scores[start:end] = chunk_scores.astype(np.float32)
        print(f"Scored unlabeled pairs: {end}/{len(unlabeled_pairs)}")
    return scores


def select_low_score_pairs(
    unlabeled_pairs: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if top_k >= len(unlabeled_pairs):
        order = np.argsort(scores)
    else:
        candidate_idx = np.argpartition(scores, top_k - 1)[:top_k]
        order = candidate_idx[np.argsort(scores[candidate_idx])]
    return unlabeled_pairs[order], scores[order]


def load_name_maps(lnc_index_path: str, drug_index_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    lnc_df = pd.read_csv(lnc_index_path)
    drug_df = pd.read_csv(drug_index_path)
    lnc_id_to_name = dict(zip(lnc_df["Index"], lnc_df["lncRNA"]))
    drug_id_to_name = dict(zip(drug_df["Index"], drug_df["Drug"]))
    return lnc_id_to_name, drug_id_to_name


def pair_to_record(
    lnc_idx: int,
    drug_idx_raw: int,
    num_lnc: int,
    lnc_id_to_name: Dict[int, str],
    drug_id_to_name: Dict[int, str],
    label: bool,
    score: float | None = None,
) -> Dict[str, object]:
    record = {
        "instruction": "Given an lncRNA and a drug, please determine whether they are associated. Respond True or False.",
        "input": f"The input pair:\n( {lnc_id_to_name.get(lnc_idx, f'lncRNA_{lnc_idx}')}, {drug_id_to_name.get(drug_idx_raw, f'drug_{drug_idx_raw}')} )",
        "output": "True" if label else "False",
        "embedding_ids": [int(lnc_idx), int(num_lnc + drug_idx_raw)],
    }
    if score is not None:
        record["rf_positive_probability"] = float(score)
    return record


def export_records(
    pos_pairs: np.ndarray,
    selected_neg_pairs: np.ndarray,
    selected_neg_scores: np.ndarray,
    num_lnc: int,
    lnc_id_to_name: Dict[int, str],
    drug_id_to_name: Dict[int, str],
    negative_output_path: str,
    balanced_output_path: str,
) -> None:
    negative_records = [
        pair_to_record(
            lnc_idx=int(pair[0]),
            drug_idx_raw=int(pair[1]),
            num_lnc=num_lnc,
            lnc_id_to_name=lnc_id_to_name,
            drug_id_to_name=drug_id_to_name,
            label=False,
            score=float(score),
        )
        for pair, score in zip(selected_neg_pairs, selected_neg_scores)
    ]

    positive_records = [
        pair_to_record(
            lnc_idx=int(pair[0]),
            drug_idx_raw=int(pair[1]),
            num_lnc=num_lnc,
            lnc_id_to_name=lnc_id_to_name,
            drug_id_to_name=drug_id_to_name,
            label=True,
        )
        for pair in pos_pairs
    ]

    balanced_records = positive_records + negative_records
    random.shuffle(balanced_records)

    with open(negative_output_path, "w", encoding="utf-8") as f:
        json.dump(negative_records, f, indent=2, ensure_ascii=False)

    with open(balanced_output_path, "w", encoding="utf-8") as f:
        json.dump(balanced_records, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select reliable negative lncRNA-drug pairs using Random Forest.")
    parser.add_argument("--data-root", type=str, default="./data/D-lnc_with_features")
    parser.add_argument("--adj-path", type=str, default="")
    parser.add_argument("--lnc-features-path", type=str, default="")
    parser.add_argument("--drug-features-path", type=str, default="")
    parser.add_argument("--lnc-index-path", type=str, default="")
    parser.add_argument("--drug-index-path", type=str, default="")
    parser.add_argument("--negative-output-path", type=str, default="./data/D-lnc_with_features/rf_selected_negative_pairs.json")
    parser.add_argument("--balanced-output-path", type=str, default="./data/D-lnc_with_features/lnc_drug_dataset_rf_balanced.json")
    parser.add_argument("--summary-output-path", type=str, default="./data/D-lnc_with_features/rf_negative_sampling_summary.json")
    parser.add_argument("--model-output-path", type=str, default="./data/D-lnc_with_features/rf_negative_sampler.joblib")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--initial-negative-multiplier", type=int, default=3)
    parser.add_argument("--provisional-negative-multiplier", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--score-chunk-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    adj_path = args.adj_path or os.path.join(args.data_root, "adj_matrix.txt")
    lnc_features_path = args.lnc_features_path or os.path.join(args.data_root, "lnc_features.txt")
    drug_features_path = args.drug_features_path or os.path.join(args.data_root, "drug_features.txt")
    lnc_index_path = args.lnc_index_path or os.path.join(args.data_root, "lncRNA_index.csv")
    drug_index_path = args.drug_index_path or os.path.join(args.data_root, "drug_index.csv")

    adj_matrix = np.loadtxt(adj_path, delimiter="\t", dtype=np.int8)
    lnc_features = load_feature_matrix(lnc_features_path)
    drug_features = load_feature_matrix(drug_features_path)

    num_lnc, num_drug = adj_matrix.shape
    pos_pairs = np.argwhere(adj_matrix == 1).astype(np.int32)
    unlabeled_pairs = np.argwhere(adj_matrix == 0).astype(np.int32)

    print(f"num_lnc={num_lnc}, num_drug={num_drug}")
    print(f"positive pairs={len(pos_pairs)}, unlabeled pairs={len(unlabeled_pairs)}")

    lnc_embeddings, drug_embeddings = fit_pca_features(lnc_features, drug_features, args.pca_dim)
    pair_feature_dim = lnc_embeddings.shape[1] * 4
    print(f"reduced lnc dim={lnc_embeddings.shape[1]}, reduced drug dim={drug_embeddings.shape[1]}")
    print(f"pair feature dim={pair_feature_dim}")

    initial_neg_size = min(len(unlabeled_pairs), len(pos_pairs) * args.initial_negative_multiplier)
    initial_neg_idx = np.random.choice(len(unlabeled_pairs), size=initial_neg_size, replace=False)
    initial_neg_pairs = unlabeled_pairs[initial_neg_idx]
    print(f"initial negative seed size={len(initial_neg_pairs)}")

    rf_stage1 = train_rf(
        pos_pairs=pos_pairs,
        neg_pairs=initial_neg_pairs,
        lnc_embeddings=lnc_embeddings,
        drug_embeddings=drug_embeddings,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_seed=args.seed,
    )

    stage1_scores = score_unlabeled_pairs(
        rf=rf_stage1,
        unlabeled_pairs=unlabeled_pairs,
        lnc_embeddings=lnc_embeddings,
        drug_embeddings=drug_embeddings,
        chunk_size=args.score_chunk_size,
    )

    provisional_size = min(len(unlabeled_pairs), len(pos_pairs) * args.provisional_negative_multiplier)
    provisional_neg_pairs, provisional_neg_scores = select_low_score_pairs(
        unlabeled_pairs=unlabeled_pairs,
        scores=stage1_scores,
        top_k=provisional_size,
    )
    print(
        "stage1 provisional negatives selected="
        f"{len(provisional_neg_pairs)}, max positive probability={float(np.max(provisional_neg_scores)):.6f}"
    )

    rf_stage2 = train_rf(
        pos_pairs=pos_pairs,
        neg_pairs=provisional_neg_pairs,
        lnc_embeddings=lnc_embeddings,
        drug_embeddings=drug_embeddings,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_seed=args.seed + 1,
    )

    final_scores = score_unlabeled_pairs(
        rf=rf_stage2,
        unlabeled_pairs=unlabeled_pairs,
        lnc_embeddings=lnc_embeddings,
        drug_embeddings=drug_embeddings,
        chunk_size=args.score_chunk_size,
    )

    selected_neg_pairs, selected_neg_scores = select_low_score_pairs(
        unlabeled_pairs=unlabeled_pairs,
        scores=final_scores,
        top_k=len(pos_pairs),
    )
    print(
        "final negatives selected="
        f"{len(selected_neg_pairs)}, max positive probability={float(np.max(selected_neg_scores)):.6f}"
    )

    lnc_id_to_name, drug_id_to_name = load_name_maps(lnc_index_path, drug_index_path)
    os.makedirs(os.path.dirname(args.negative_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.balanced_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)

    export_records(
        pos_pairs=pos_pairs,
        selected_neg_pairs=selected_neg_pairs,
        selected_neg_scores=selected_neg_scores,
        num_lnc=num_lnc,
        lnc_id_to_name=lnc_id_to_name,
        drug_id_to_name=drug_id_to_name,
        negative_output_path=args.negative_output_path,
        balanced_output_path=args.balanced_output_path,
    )

    summary = RFSamplingSummary(
        num_lnc=int(num_lnc),
        num_drug=int(num_drug),
        num_positive_pairs=int(len(pos_pairs)),
        num_unlabeled_pairs=int(len(unlabeled_pairs)),
        num_selected_negatives=int(len(selected_neg_pairs)),
        pair_feature_dim=int(pair_feature_dim),
        lnc_pca_dim=int(lnc_embeddings.shape[1]),
        drug_pca_dim=int(drug_embeddings.shape[1]),
        initial_negative_multiplier=int(args.initial_negative_multiplier),
        provisional_negative_multiplier=int(args.provisional_negative_multiplier),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth) if args.max_depth is not None else None,
        random_seed=int(args.seed),
    )

    with open(args.summary_output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    joblib.dump(
        {
            "rf_stage1": rf_stage1,
            "rf_stage2": rf_stage2,
            "lnc_embeddings": lnc_embeddings,
            "drug_embeddings": drug_embeddings,
            "selected_negative_pairs": selected_neg_pairs,
            "selected_negative_scores": selected_neg_scores,
        },
        args.model_output_path,
    )

    print(f"Saved selected negatives to {args.negative_output_path}")
    print(f"Saved balanced dataset to {args.balanced_output_path}")
    print(f"Saved summary to {args.summary_output_path}")
    print(f"Saved RF artifacts to {args.model_output_path}")


if __name__ == "__main__":
    main()
