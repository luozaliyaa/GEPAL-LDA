import os

from torch import nn
from tqdm import tqdm
from model import MultiScaleAdapter, EnhancedPromptAdapter, GCNModel, GraphSAGEModel, GATModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import json
import torch
import transformers
from safetensors.torch import load_file
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizerFast as LlamaTokenizer
from safetensors.torch import load_file
from load_lnc_drug_data import build_gat_data
from preTrain import Node2Prefix, get_device
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sns


try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None
base_path = '' #base_model /path/to/model


prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given an lncRNA and a drug, please determine whether they are associated. Respond True or False.

### Input:
{}

### Response:

"""


def load_test_dataset(path):
    test_dataset = json.load(open(path, "r"))
    return test_dataset


def inference(fold: int = 0,
              lora_weights: str = "",
              test_data_path: str = "",
              ckpt_path: str = "",
              metrics_log_path: str = "",
              plot_path: str = ""
              ):
    cuda = "cuda:0"
    device = get_device(prefer_gpu=0)
    # lora_weights = f"./train_res/cMap/fold_{fold}/lora-alpaca"
    # test_data_path = f"./data/D-lnc_with_features/cMap/10fold/fold_{fold}/lnc_drug_test.json"
    # ckpt_path = f"./train_res/cMap/fold_{fold}/gat_adapter.pth"
    data_obj = build_gat_data(fold=fold)
    node_features = data_obj.x.to(device)
    edge_index = data_obj.edge_index.to(device)
    test_dataset = load_test_dataset(test_data_path)

    tokenizer = LlamaTokenizer.from_pretrained(base_path)

    model = LlamaForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16
    ).to(device)
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     torch_dtype=torch.float16,
    # ).to(device)

    # 1. 加载 LoRA 配置
    peft_config = PeftConfig.from_pretrained(lora_weights, local_files_only=True)

    # 2. 包装 base model
    model = get_peft_model(model, peft_config)

    # 3. 加载 LoRA 权重
    weight_file = os.path.join(lora_weights, "adapter_model.safetensors")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Expected {weight_file} but not found.")

    print(f"Loading LoRA weights from {weight_file}")
    weights = load_file(weight_file)  # safetensors 返回一个 state_dict
    set_peft_model_state_dict(model, weights)

    model = model.to(device)
    dtype = model.dtype
    gat = GATModel(in_dim=256, out_dim=128).to(device)
    gcn = GCNModel(in_dim=256, out_dim=128).to(device)
    sage = GraphSAGEModel(in_dim=256, out_dim=128).to(device)
    # adapter = Node2Prefix(input_dim=128 * 2, llm_dim=model.config.hidden_size, num_prefix=6).to(device)
    proj = nn.Linear(2 * 128, 128).to(device)
    adapter = MultiScaleAdapter(input_dim=128 * 2,
                                llm_dim=model.config.hidden_size,
                                num_prefix=6,
                                hidden_dim=512).to(device)

    # 如果你有 lnc_adapter / drug_adapter，也要定义
    lnc_adapter = MultiScaleAdapter(
        input_dim=128,
        llm_dim=model.config.hidden_size,
        num_prefix=3,  # num_prefix // 2
        hidden_dim=512
    ).to(device)

    drug_adapter = MultiScaleAdapter(
        input_dim=128,
        llm_dim=model.config.hidden_size,
        num_prefix=3,  # num_prefix // 2
        hidden_dim=512
    ).to(device)

    diff_adapter = MultiScaleAdapter(
        input_dim=128,
        num_prefix=3,
        llm_dim=model.config.hidden_size,
        hidden_dim=512
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if "gat" in state:
        gat.load_state_dict(state["gat"])
    else:
        print("Warning: 'gat' not found in state dict")

    if "gcn" in state:
        gcn.load_state_dict(state["gcn"])
    else:
        print("Warning: 'gcn' not found in state dict")

    if "sage" in state:
        sage.load_state_dict(state["sage"])
    else:
        print("Warning: 'sage' not found in state dict")

    if "node2prefix" in state:
        adapter.load_state_dict(state["node2prefix"])
    else:
        print("Warning: 'node2prefix' not found in state dict")

    if "alpha" in state:
        alpha = state["alpha"]  # tensor，需要赋值到模型里
    else:
        print("Warning: 'alpha' not found in state dict")

    if "proj" in state:
        proj.load_state_dict(state["proj"])
    else:
        print("Warning: 'node2prefix' not found in state dict")

    if "lnc_adapter" in state:
        lnc_adapter.load_state_dict(state["lnc_adapter"])
    else:
        print("Warning: 'lnc_adapter' not found in state dict")

    if "drug_adapter" in state:
        drug_adapter.load_state_dict(state["drug_adapter"])
    else:
        print("Warning: 'drug_adapter' not found in state dict")

    if "diff_adapter" in state:
        diff_adapter.load_state_dict(state["diff_adapter"])
    else:
        print("Warning: 'diff_adapter' not found in state dict")

    # --- 前向传播 ---
    gat_out = gat(node_features, edge_index)  # [N, out_dim]
    gcn_out = gcn(node_features, edge_index)  # [N, out_dim]
    sage_out = sage(node_features, edge_index)

    fused_out = torch.cat([gat_out, gcn_out], dim=-1)
    fused_out = proj(fused_out)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model = model.eval()

    # 定义存储变量
    y_true = []
    y_pred = []
    y_score = []

    # True / False 对应的 token id
    true_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]

    # === 评估指标 ===
    from datetime import datetime

    log_path = metrics_log_path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f_log:
        for i, data in enumerate(tqdm(test_dataset, desc="Predicting lncRNA-drug associations")):
            try:
                ent = data["input"]
                ans = data["output"]
                ids = torch.LongTensor(data["embedding_ids"]).reshape(1, -1).to(device)

                # === 图特征提取 ===
                lnc_ids, drug_ids = ids[:, 0], ids[:, 1]  # [batch]
                lnc_feats = fused_out[lnc_ids]  # [batch, out_dim]
                drug_feats = fused_out[drug_ids]  # [batch, out_dim]
                # lnc_feats = sage_out[lnc_ids]
                # drug_feats = sage_out[drug_ids]

                # strategy one
                pair_feats = torch.cat([lnc_feats, drug_feats], dim=-1)
                prefix1 = adapter(pair_feats).to(device=device, dtype=model.dtype)

                # strategy two 分别生成 prefix
                prefix_lnc = lnc_adapter(lnc_feats).to(device=device,dtype=model.dtype)  # [batch, num_prefix//2, llm_dim]
                prefix_drug = drug_adapter(drug_feats).to(device=device,dtype=model.dtype)  # [batch, num_prefix//2, llm_dim]
                prefix2 = torch.cat([prefix_lnc, prefix_drug], dim=1)  # [batch, num_prefix, llm_dim]

                # strategy three
                diff_feats = torch.abs(lnc_feats - drug_feats)  # [batch, out_dim]
                prefix3 = diff_adapter(diff_feats).to(device=device, dtype=model.dtype)

                # 拼接 prefix

                # prefix = torch.cat([prefix2, prefix3], dim=1)
                prefix = prefix1

                # === 构造输入 ===
                prompt = prompt_template.format(ent)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                token_embeds = model.model.model.embed_tokens(input_ids)
                input_embeds = torch.cat((prefix, token_embeds), dim=1)

                # === 生成预测 ===
                generate_ids = model.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=16,
                    pad_token_id=tokenizer.pad_token_id
                )

                context = \
                tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                response = \
                tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                response = response.replace(context, "").strip()

                if response == "":
                    response = "False"

                # === 计算概率 ===
                with torch.no_grad():
                    outputs = model(inputs_embeds=input_embeds, return_dict=True)
                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    prob_true = probs[0, true_id].item()
                    prob_false = probs[0, false_id].item()

                # === 保存结果 ===
                y_true.append(1 if "True" in ans else 0)
                y_pred.append(1 if "True" in response else 0)
                y_score.append(prob_true)

                lnc_id = int(lnc_ids.item())
                drug_id = int(drug_ids.item())

                # === 日志输出 ===
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                is_mismatch = (response == "True" and "False" in ans)
                flag = "⚠️" if is_mismatch else "✅"

                log_line = (
                    f"[{timestamp}] [{i + 1}] {flag} "
                    f"lnc_id={lnc_id} | drug_id={drug_id} | "
                    f"response={response} | answer={ans} |  "
                    f"true_score={prob_true:.4f} | false_score={prob_false:.4f}"
                )

                f_log.write(log_line + "\n")
                f_log.flush()

                # === 定期打印中间指标 ===
                if (i + 1) % 500 == 0:
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    try:
                        auc_val = roc_auc_score(y_true, y_score)
                    except ValueError:
                        auc_val = float("nan")
                    msg = f"[{timestamp}] Step {i + 1}: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc_val:.4f}"

                    f_log.write(msg + "\n")
                    f_log.flush()

            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                err_msg = f"[{timestamp}] ⚠️ Error on sample {i + 1}: {str(e)}"

                f_log.write(err_msg + "\n")
                f_log.flush()

        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc_val = roc_auc_score(y_true, y_score)  # 改名
        pr_auc_val = average_precision_score(y_true, y_score)  # 改名

        print("ACC:", acc)
        print("Precision:", p)
        print("Recall:", r)
        print("F1:", f1)
        print("AUC (ROC):", roc_auc_val)
        print("AUPR (PR):", pr_auc_val)

        f_log.write("\n=== Final Evaluation Results ===\n")
        f_log.write(f"ACC: {acc:.4f}\n")
        f_log.write(f"Precision: {p:.4f}\n")
        f_log.write(f"Recall: {r:.4f}\n")
        f_log.write(f"F1: {f1:.4f}\n")
        f_log.write(f"AUC (ROC): {roc_auc_val:.4f}\n")
        f_log.write(f"AUPR (PR): {pr_auc_val:.4f}\n")
        f_log.flush()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_curve_auc = auc(fpr, tpr)  # sklearn.metrics.auc，不会被覆盖
    # ---------------------- PR 曲线 ----------------------
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_curve_auc = auc(recall, precision)

    plot_log_path = plot_path
    os.makedirs(plot_log_path, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_curve_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_log_path, "roc_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"PR curve (AUPR = {pr_curve_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_log_path, "pr_curve.png"), dpi=300)
    plt.close()

    return {
        "ACC": acc,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "ROC_AUC": roc_auc_val,
        "PR_AUC": pr_auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision,
        "recall_curve": recall
    }

# if __name__ == "__main__":
#     inference()