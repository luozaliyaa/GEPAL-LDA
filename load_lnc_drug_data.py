import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import Data


import numpy as np
import torch
from torch_geometric.data import Data
import json
from experiment_paths import DATA_ROOT, FOLD_ROOT

def build_gat_data(d=256, fold=1):
    # === 1. 读取特征和邻接矩阵 ===
    lnc_features = np.loadtxt(DATA_ROOT / "lnc_features.txt", delimiter="\t")
    drug_features = np.loadtxt(DATA_ROOT / "drug_features.txt", delimiter="\t")
    adj_matrix = np.loadtxt(DATA_ROOT / "adj_matrix.txt", delimiter="\t").astype(int)
    test_json = FOLD_ROOT / f"fold_{fold}" / "lnc_drug_test.json"
    num_lnc, num_drug = adj_matrix.shape
    print(f"lncRNA: {num_lnc}, drug: {num_drug}")

    # === 2. PCA降维 ===
    from sklearn.decomposition import PCA
    lnc_proj = PCA(n_components=d).fit_transform(lnc_features)
    drug_proj = PCA(n_components=d).fit_transform(drug_features)
    # lnc_proj = lnc_features
    # drug_proj = drug_features


    X = torch.tensor(np.vstack([lnc_proj, drug_proj]), dtype=torch.float)
    print(f"Node feature shape: {X.shape}")  # [num_lnc+num_drug, d]

    # === 3. 邻接矩阵 → edge_index ===
    lnc_ids, drug_ids = np.where(adj_matrix == 1)
    drug_ids = drug_ids + num_lnc
    edge_index = np.vstack([np.concatenate([lnc_ids, drug_ids]),
                            np.concatenate([drug_ids, lnc_ids])])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    print(f"Total edges before removal: {edge_index.shape[1]}")

    # === 4. 读取测试集正样本并删除 ===
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_pos_pairs = set()
    for item in test_data:
        if item["output"].strip().lower() == "true":
            lnc_id, drug_id = item["embedding_ids"]  # 已经偏移好的编号
            test_pos_pairs.add((lnc_id, drug_id))
            test_pos_pairs.add((drug_id, lnc_id))  # 双向边

    print(f"Test set positive pairs to remove: {len(test_pos_pairs)//2}")

    # 删除测试集边
    mask = [tuple(edge) not in test_pos_pairs for edge in edge_index.t().tolist()]
    edge_index_filtered = edge_index[:, mask]
    print(f"Edges after removal: {edge_index_filtered.shape[1]}")

    # === 5. 构建 Data 对象 ===
    data = Data(x=X, edge_index=edge_index_filtered)
    # data = Data(x=X, edge_index=edge_index)
    print("Data construction done (test edges removed)")

    return data


