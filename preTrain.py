
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModelForCausalLM
# -------------------------------
# 1. GAT + Adapter 模型
# -------------------------------
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, f1_score
from load_lnc_drug_data import build_gat_data
import numpy as np
from tqdm import tqdm

def get_device(prefer_gpu: int = 0):
    """
    获取计算设备，优先使用 GPU。
    参数 prefer_gpu: 想用的逻辑 GPU id（默认为0）。
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if prefer_gpu < num_gpus:
            print(f"Using GPU {prefer_gpu}/{num_gpus}: {torch.cuda.get_device_name(prefer_gpu)}")
            return torch.device(f"cuda:{prefer_gpu}")
        else:
            print(f"Requested GPU {prefer_gpu}, but only {num_gpus} GPUs available. Falling back to cuda:0")
            return torch.device("cuda:0")
    else:
        print("No GPU available, using CPU.")
        return torch.device("cpu")

class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(in_channels=hidden_dim*heads, out_channels=out_dim, heads=1, concat=True, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

class MLPAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPAdapter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class LncDrugDataset(Dataset):
    def __init__(self, lnc_idx, drug_idx, labels, node_features, edge_index):
        self.lnc_idx = lnc_idx
        self.drug_idx = drug_idx
        self.labels = labels
        self.node_features = node_features
        self.edge_index = edge_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "graph": {
                "x": self.node_features,
                "edge_index": self.edge_index,
                "lnc_idx": self.lnc_idx[idx],
                "drug_idx": self.drug_idx[idx],
            },
            "label": self.labels[idx],
        }

def collate_fn(batch):
    """
    batch: list of samples from LncDrugDataset
    每个 sample 结构：
    {
        "graph": {
            "x": node_features,
            "edge_index": edge_index,
            "lnc_idx": int,
            "drug_idx": int
        },
        "label": int
    }
    """
    graph = {
        "x": batch[0]["graph"]["x"],  # 所有 sample 共享同一个节点特征
        "edge_index": batch[0]["graph"]["edge_index"],  # 所有 sample 共享同一张图
        "lnc_idx": torch.tensor([item["graph"]["lnc_idx"] for item in batch], dtype=torch.long),
        "drug_idx": torch.tensor([item["graph"]["drug_idx"] for item in batch], dtype=torch.long),
    }

    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {"graph": graph, "label": labels}

class Node2Prefix(nn.Module):
    def __init__(self, input_dim: int, llm_dim: int, num_prefix: int = 1):
        """
        input_dim: GAT 输出维度
        llm_dim: LLM embedding 维度（通常 4096 或 5120）
        num_prefix: 每个样本 prefix token 数量
        """
        super().__init__()
        self.num_prefix = num_prefix
        self.adapter = nn.Linear(input_dim, llm_dim * num_prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim] (GAT 输出)
        return: [batch_size, num_prefix, llm_dim]
        """
        batch_size = x.size(0)
        prefix = self.adapter(x)  # [batch, llm_dim * num_prefix]
        prefix = prefix.view(batch_size, self.num_prefix, -1)
        # return prefix
        return prefix.to(x.dtype)


def cross_val_train(
    lnc_idx, drug_idx, labels, node_features, edge_index, device,
    base_model_name="/dev/shm/MMed-Llama-3-8B",
    k_folds=5, batch_size=64, num_epochs=3
):
    dataset = LncDrugDataset(lnc_idx, drug_idx, labels, node_features, edge_index)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # LLaMA 没有 pad_token，用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.float16, local_files_only=True
    ).to(device)
    for param in llm.parameters():
        param.requires_grad = False

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold = 0
    fold_metrics = []  # 每个 fold 存储 {acc, f1, loss}

    for train_idx, val_idx in kf.split(range(len(dataset))):
        fold += 1
        print(f"=== Fold {fold} ===")

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        gat = GATModel(in_dim=node_features.shape[1], out_dim=128).to(device)
        # adapter = MLPAdapter(in_dim=128*2, out_dim=256).to(device)
        adapter = Node2Prefix(input_dim=128 * 2, llm_dim=llm.config.hidden_size, num_prefix=5).to(device)
        optimizer = torch.optim.Adam(list(gat.parameters()) + list(adapter.parameters()), lr=1e-3)

        for epoch in range(num_epochs):
            # 训练
            gat.train()
            adapter.train()
            total_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
            for batch in train_pbar:
                graph = batch["graph"]
                labels_batch = batch["label"].to(device)

                x = graph["x"].to(device)
                edge_index = graph["edge_index"].to(device)
                lnc_idx_batch = graph["lnc_idx"].to(device)
                drug_idx_batch = graph["drug_idx"].to(device)

                node_emb = gat(x, edge_index)
                lnc_emb = node_emb[lnc_idx_batch]
                drug_emb = node_emb[drug_idx_batch]
                pair_emb = torch.cat([lnc_emb, drug_emb], dim=-1)

                prefix_emb = adapter(pair_emb)

                # 构造文本 prompt
                # 这是一对lncRNA与药物经过GAT得到的特征向量，请你判断他们两个之间存在关联的概率，从0到1
                prompt_texts = ["这是一对lncRNA与药物经过GAT得到的特征向量，请你判断他们两个之间是否存在关联。回答 Associated 或 Not Associated"] * len(labels_batch)
                tokenized = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)

                # 直接用 inputs_embeds 拼接 Adapter embedding
                # token_embeds = llm.model.embed_tokens(tokenized["input_ids"])  # [batch, seq_len, hidden]
                token_embeds = llm.get_input_embeddings()(tokenized["input_ids"])
                input_embeds = torch.cat([prefix_emb, token_embeds], dim=1).to(llm.dtype)  # [batch, num_prefix+seq_len, hidden]

                # attention mask 扩展
                prefix_mask = torch.ones(prefix_emb.size()[:-1], device=device, dtype=torch.long)
                attention_mask = torch.cat([prefix_mask, tokenized["attention_mask"]], dim=1)

                outputs = llm(
                    input_ids=None,
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True
                )

                logits = outputs.logits

                associated_ids = tokenizer.encode("Associated", add_special_tokens=False)
                not_associated_ids = tokenizer.encode("Not Associated", add_special_tokens=False)

                # 只取第一个 token 作为近似
                associated_id = associated_ids[0]
                not_associated_id = not_associated_ids[0]

                # 2. logits_first: prefix 之后第一个 token 的 logits
                target_pos = prefix_emb.size(1)
                logits_first = logits[:, target_pos, :]  # [batch, vocab_size]

                # 3. 选取正负类 logits
                logits_selected = torch.stack([
                    logits_first[:, associated_id],  # 正类
                    logits_first[:, not_associated_id]  # 负类
                ], dim=1)  # [batch, 2]

                # 4. 交叉熵 loss
                labels_batch_ce = labels_batch.long()  # 0=Not Associated, 1=Associated
                loss = F.cross_entropy(logits_selected, labels_batch_ce)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)

            # 验证
            gat.eval()
            adapter.eval()
            all_preds, all_labels = [], []
            val_pbar = tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    graph = batch["graph"]
                    labels_batch = batch["label"].to(device)

                    x = graph["x"].to(device)
                    edge_index = graph["edge_index"].to(device)
                    lnc_idx_batch = graph["lnc_idx"].to(device)
                    drug_idx_batch = graph["drug_idx"].to(device)

                    # 1. GAT 提取节点表示
                    node_emb = gat(x, edge_index)
                    lnc_emb = node_emb[lnc_idx_batch]
                    drug_emb = node_emb[drug_idx_batch]
                    pair_emb = torch.cat([lnc_emb, drug_emb], dim=-1)

                    # 2. Adapter → Prefix embedding
                    prefix_emb = adapter(pair_emb)  # [batch, num_prefix, hidden]

                    # 3. 构造文本 prompt
                    prompt_texts = ["这是一对lncRNA与药物经过GAT得到的特征向量，请判断该lncRNA与药物是否有关联。回答 Associated 或 Not Associated"] * len(
                        labels_batch)
                    tokenized = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)

                    # 4. 获取 LLM token embedding
                    token_embeds = llm.get_input_embeddings()(tokenized["input_ids"])  # [batch, seq_len, hidden]
                    input_embeds = torch.cat([prefix_emb, token_embeds], dim=1).to(llm.dtype) # [batch, num_prefix+seq_len, hidden]

                    # 5. Attention mask
                    prefix_mask = torch.ones(prefix_emb.size()[:-1], device=device)
                    attention_mask = torch.cat([prefix_mask, tokenized["attention_mask"]], dim=1)

                    # 6. 前向传播
                    outputs = llm(
                        input_ids=None,
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    logits = outputs.logits  # [batch, seq_len, vocab_size]


                    associated_ids = tokenizer.encode("Associated", add_special_tokens=False)
                    not_associated_ids = tokenizer.encode("Not Associated", add_special_tokens=False)

                    # 2. 取第一个 token 作为近似
                    associated_id = associated_ids[0]
                    not_associated_id = not_associated_ids[0]

                    # 3. logits_first: prefix 之后第一个 token 的 logits
                    target_pos = prefix_emb.size(1)
                    logits_first = logits[:, target_pos, :]  # [batch, vocab_size]

                    # 4. 选出正负类 logits
                    logits_selected = torch.stack([
                        logits_first[:, associated_id],  # 正类
                        logits_first[:, not_associated_id]  # 负类
                    ], dim=1)  # [batch, 2]

                    # 5. softmax + argmax
                    probs = torch.softmax(logits_selected, dim=-1)
                    preds = torch.argmax(probs, dim=1).cpu().tolist()

                    # 11. 保存结果
                    all_preds.extend(preds)
                    all_labels.extend(labels_batch.cpu().tolist())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            print(f"Fold {fold} Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
            # 在每个 fold 完成后 append
            fold_metrics.append({
                "fold": fold,
                "loss": avg_loss,
                "val_acc": acc,
                "val_f1": f1
            })

        print(f"Fold {fold} finished.\n")

    # 保存最后的模型
    torch.save({
        "gat_state_dict": gat.state_dict(),
        "adapter_state_dict": adapter.state_dict()
    }, "gat_adapter_final.pth")
    print("model save to gat_adapter_final.pth")

    # 保存 fold_metrics 到 txt 文件
    with open("fold_metrics.txt", "w") as f:
        # 写表头
        f.write("fold\tloss\tval_acc\tval_f1\n")
        for m in fold_metrics:
            f.write(f"{m['fold']}\t{m['loss']:.4f}\t{m['val_acc']:.4f}\t{m['val_f1']:.4f}\n")

    print("Fold metrics saved to fold_metrics.txt")

    # 最后返回
    return fold_metrics


# -------------------------------
# 5. 使用示例
# -------------------------------
if __name__ == "__main__":

    device = get_device(prefer_gpu=0)
    print("Using device:", device)

    # adj_matrix = np.loadtxt("./data/raw/lnc_drug_adj_matrix.txt", delimiter="\t").astype(int)
    adj_matrix = np.loadtxt("./data/D-lnc/adj_matrix.txt", delimiter="\t").astype(int)
    num_lnc, num_drug = adj_matrix.shape
    # # 2. 转为 PyTorch 张量
    # adj_matrix = torch.tensor(adj_matrix, dtype=torch.int)
    print("adj matrix ", adj_matrix.shape)
    data_obj = build_gat_data()

    # 1️⃣ 获取正负样本索引
    pos_pairs = [(i, j) for i in range(num_lnc) for j in range(num_drug) if adj_matrix[i, j] == 1]
    neg_pairs = [(i, j) for i in range(num_lnc) for j in range(num_drug) if adj_matrix[i, j] == 0]

    # 2️⃣ 随机采样负样本，使数量和正样本相等
    neg_pairs_sampled = random.sample(neg_pairs, len(pos_pairs))

    # 3️⃣ 合并正负样本
    all_pairs = pos_pairs + neg_pairs_sampled

    # 4️⃣ 生成训练列表
    lnc_idx_list = [i for i, j in all_pairs]
    drug_idx_list = [num_lnc + j for i, j in all_pairs]  # 药物节点偏移
    labels_list = [1] * len(pos_pairs) + [0] * len(neg_pairs_sampled)

    print(f"Total samples: {len(labels_list)} | Positive: {len(pos_pairs)} | Negative: {len(neg_pairs_sampled)}")

    results = cross_val_train(
        lnc_idx=lnc_idx_list,
        drug_idx=drug_idx_list,
        labels=labels_list,
        node_features=data_obj.x,
        edge_index=data_obj.edge_index,
        device=device,
        k_folds=5,
        batch_size=64,
        num_epochs=3
    )
