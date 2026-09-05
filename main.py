

from my_finetune import train
from my_inference import inference
from experiment_paths import FOLD_ROOT
import torch, gc
import json
import os


def clear_gpu_memory():
    """安全释放显存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    NUM_FOLDS = 10
    all_metrics = []

    for fold in range(4,5):
        print("=" * 60)
        print(f"▶️ 开始第 {fold} 折训练")
        print("=" * 60)

        # # === 1️⃣ 训练 ===
        train(
            fold=fold,
            num_epochs=2,
            data_path=str(FOLD_ROOT / f"fold_{fold}" / "lnc_drug_train.json"),
            val_data_path=str(FOLD_ROOT / f"fold_{fold}" / "lnc_drug_val.json"),
            output_dir=f"./train_res_rf/fold_{fold}/lora-alpaca",
            model_save_path = f"./train_res_rf/fold_{fold}/gat_adapter.pth"
        )

        print(f"✅ 第 {fold} 折训练完成！")

        # === 2️⃣ 推理 / 评估 ===
        print(f"▶️ 开始第 {fold} 折推理")
        metrics = inference(
            fold=fold,
            lora_weights=f"./train_res_rf/fold_{fold}/lora-alpaca",
            test_data_path=str(FOLD_ROOT / f"fold_{fold}" / "lnc_drug_test.json"),
            ckpt_path=f"./train_res_rf/fold_{fold}/gat_adapter.pth",
            metrics_log_path=f"./metrics_rf/fold_{fold}/predict_log_fold_{fold}.txt",
            plot_path=f"./metrics_rf/fold_{fold}/fold_{fold}_plots"
        )
        all_metrics.append(metrics)
        print(f"✅ 第 {fold} 折推理完成！\n")

    import numpy as np
    import json
    import os

    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_jsonable(v) for v in obj]
        return obj

    # 保存 metrics
    metrics_path = f"./metrics_rf/10fold/fold_{fold}_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics), f, indent=2, ensure_ascii=False)
    print(f"✅ Saved metrics for fold {fold} to {metrics_path}")


    # === 统计十折指标均值 ± 标准差 ===
    # def get_stat(name):
    #     values = [m[name] for m in all_metrics]
    #     return np.mean(values), np.std(values)

    # acc_mean, acc_std = get_stat("ACC")
    # p_mean, p_std = get_stat("Precision")
    # r_mean, r_std = get_stat("Recall")
    # f1_mean, f1_std = get_stat("F1")
    # roc_auc_mean, roc_auc_std = get_stat("ROC_AUC")
    # pr_auc_mean, pr_auc_std = get_stat("PR_AUC")

    # print("\n=== 10-Fold Cross-Validation Results ===")
    # print(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    # print(f"Precision: {p_mean:.4f} ± {p_std:.4f}")
    # print(f"Recall: {r_mean:.4f} ± {r_std:.4f}")
    # print(f"F1: {f1_mean:.4f} ± {f1_std:.4f}")
    # print(f"ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}")
    # print(f"PR AUC: {pr_auc_mean:.4f} ± {pr_auc_std:.4f}")

    # # === 绘制十折 ROC 曲线 ===
    # plt.figure()
    # for i, m in enumerate(all_metrics):
    #     plt.plot(m["fpr"], m["tpr"], lw=1, alpha=0.6, label=f"Fold {i+1} (AUC={m['ROC_AUC']:.4f})")
    # plt.plot([0, 1], [0, 1], 'k--', lw=1)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("10-Fold ROC Curves")
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("./metrics/roc_curve_10fold.png", dpi=300)
    # plt.close()

    # # === 绘制十折 PR 曲线 ===
    # plt.figure()
    # for i, m in enumerate(all_metrics):
    #     plt.plot(m["recall_curve"], m["precision_curve"], lw=1, alpha=0.6, label=f"Fold {i+1} (AUPR={m['PR_AUC']:.4f})")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("10-Fold PR Curves")
    # plt.legend(loc="lower left")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("./metrics/pr_curve_10fold.png", dpi=300)
    # plt.close()
