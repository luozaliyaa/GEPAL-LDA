import json
import random
import numpy as np
import pandas as pd

def make_json_dataset_with_names(
    adj_matrix_path="./data/D-lnc_with_features/cMap/adj_matrix.txt",
    drug_index_path="./data/D-lnc_with_features/cMap/drug_index.csv",
    lnc_index_path="./data/D-lnc_with_features/cMap/lncRNA_index.csv",
    save_json="./data/D-lnc_with_features/cMap/lnc_drug_dataset.json",
    save_lnc_map="./data/D-lnc_with_features/cMap/lnc_id_map.txt",
    save_drug_map="./data/D-lnc_with_features/cMap/drug_id_map.txt"

):
    # 1️⃣ 读取邻接矩阵和索引表
    adj_matrix = np.loadtxt(adj_matrix_path, delimiter="\t").astype(int)
    num_lnc, num_drug = adj_matrix.shape
    print(f"lncRNA: {num_lnc}, drug: {num_drug}")

    drug_df = pd.read_csv(drug_index_path)      # 列名: Drug,Index
    lnc_df = pd.read_csv(lnc_index_path)        # 列名: lncRNA,Index

    drug_id2name = dict(zip(drug_df["Index"], drug_df["Drug"]))
    lnc_id2name = dict(zip(lnc_df["Index"], lnc_df["lncRNA"]))

    # 2️⃣ 构建正负样本
    pos_pairs = [(i, j) for i in range(num_lnc) for j in range(num_drug) if adj_matrix[i, j] == 1]
    neg_pairs = [(i, j) for i in range(num_lnc) for j in range(num_drug) if adj_matrix[i, j] == 0]

    neg_pairs_sampled = random.sample(neg_pairs, len(pos_pairs))
    all_pairs = pos_pairs + neg_pairs_sampled
    labels_list = [1] * len(pos_pairs) + [0] * len(neg_pairs_sampled)

    # 3️⃣ 生成 JSON 数据（列表形式）
    dataset = []
    instruction = "Given an lncRNA and a drug, please determine whether they are associated. Respond True or False."

    for (lnc_idx, drug_idx_raw), label in zip(all_pairs, labels_list):
        drug_idx = num_lnc + drug_idx_raw  # 偏移
        lnc_name = lnc_id2name.get(lnc_idx, f"lncRNA_{lnc_idx}")
        drug_name = drug_id2name.get(drug_idx_raw, f"drug_{drug_idx_raw}")

        item = {
            "instruction": instruction,
            "input": f"The input pair:\n( {lnc_name}, {drug_name} )",
            "output": "True" if label == 1 else "False",
            "embedding_ids": [lnc_idx, drug_idx]
        }
        dataset.append(item)

    # 保存成 JSON 数组
    with open(save_json, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} samples to {save_json}")

    # 4️⃣ 保存映射表（带真实名字）
    with open(save_lnc_map, "w") as f:
        for i in range(num_lnc):
            name = lnc_id2name.get(i, f"lncRNA_{i}")
            f.write(f"{i}\t{name}\n")

    with open(save_drug_map, "w") as f:
        for j in range(num_drug):
            global_id = num_lnc + j
            name = drug_id2name.get(j, f"drug_{j}")
            f.write(f"{global_id}\t{name}\n")

    print(f"Saved lnc_id_map.txt ({num_lnc}) and drug_id_map.txt ({num_drug})")

make_json_dataset_with_names()