import pandas as pd
import json

# === 1. 文件路径 ===
predict_csv = "predict_pairs.csv"
lnc_index_path = "./data/D-lnc_with_features/lncRNA_index.csv"
drug_index_path = "./data/D-lnc_with_features/drug_index.csv"

# === 2. 读取数据 ===
predict_df = pd.read_csv(predict_csv)
lnc_df = pd.read_csv(lnc_index_path)
drug_df = pd.read_csv(drug_index_path)

# === 3. 构建索引映射 ===
lnc_map = dict(zip(lnc_df["lncRNA"].astype(str).str.strip(), lnc_df["Index"].astype(int)))
drug_map = dict(zip(drug_df["Drug"].astype(str).str.strip(), drug_df["Index"].astype(int)))
drug_offset = len(lnc_map)

# === 4. 构建输出数据 ===
output_list = []

for _, row in predict_df.iterrows():
    lnc = str(row["lncRNA"]).strip()
    drug = str(row["Drug"]).strip()

    if lnc not in lnc_map or drug not in drug_map:
        print(f"⚠️ 跳过未找到索引的样本: {lnc}, {drug}")
        continue

    lnc_id = lnc_map[lnc]
    drug_id = drug_map[drug] + drug_offset

    record = {
        "instruction": "Given an lncRNA and a drug, please determine whether they are associated. Respond True or False.",
        "input": f"The input pair:\n( {lnc}, {drug} )",
        "output": "True",
        "embedding_ids": [lnc_id, drug_id]
    }
    output_list.append(record)

# === 5. 保存 JSON 文件 ===
out_path = "predict_pairs_formatted.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output_list, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共生成 {len(output_list)} 条样本，已保存到 {out_path}")
