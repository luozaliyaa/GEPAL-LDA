# import json
# import random

# # 固定随机种子，保证每次划分一致
# random.seed(42)

# # 读取原始数据
# with open("lnc_drug_dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 打乱数据
# random.shuffle(data)


# train_ratio = 0.80
# val_ratio = 0.10
# test_ratio = 0.10

# n_total = len(data)
# n_train = int(n_total * train_ratio)
# n_val = int(n_total * val_ratio)

# train_data = data[:n_train]
# val_data = data[n_train:n_train + n_val]
# test_data = data[n_train + n_val:]

# # 保存结果
# with open("./data/D-lnc_with_features/lnc_drug_train.json", "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=2)

# with open("./data/D-lnc_with_features/lnc_drug_val.json", "w", encoding="utf-8") as f:
#     json.dump(val_data, f, ensure_ascii=False, indent=2)

# with open("./data/D-lnc_with_features/lnc_drug_test.json", "w", encoding="utf-8") as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=2)

# print(f"total: {n_total}")
# print(f"train: {len(train_data)} , valid: {len(val_data)} , test: {len(test_data)} ")

import json
import random
import os

# 固定随机种子，保证划分可复现
random.seed(42)

# 读取原始数据
# with open("./data/D-lnc_with_features/lnc_drug_dataset.json", "r", encoding="utf-8") as f:
with open("./data/D-lnc_with_features/lnc_drug_dataset_rf_balanced.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 打乱数据
random.shuffle(data)

# === 10折参数 ===
num_folds = 10
fold_size = len(data) // num_folds

save_dir = "./data/D-lnc_with_features/10fold_rf"
os.makedirs(save_dir, exist_ok=True)

for fold in range(num_folds):
    # === 确定当前折的测试集范围 ===
    start = fold * fold_size
    end = (fold + 1) * fold_size if fold < num_folds - 1 else len(data)

    test_data = data[start:end]
    remain_data = data[:start] + data[end:]

    # 可以再把剩下的数据划分为训练和验证集（例如 9:1）
    val_ratio = 0.1
    n_val = int(len(remain_data) * val_ratio)
    val_data = remain_data[:n_val]
    train_data = remain_data[n_val:]

    # 保存到对应的文件夹
    fold_dir = os.path.join(save_dir, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    with open(os.path.join(fold_dir, "lnc_drug_train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(fold_dir, "lnc_drug_val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(fold_dir, "lnc_drug_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"[Fold {fold+1}] train={len(train_data)} val={len(val_data)} test={len(test_data)}")

print(f"\n✅ 共生成 {num_folds} 份交叉验证数据，已保存至: {save_dir}")

