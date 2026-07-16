import os
from typing import List

# import wandb
# wandb.init(mode="offline")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["WANDB_DISABLED"] = "true"


import fire
import torch
import transformers
from datasets import load_dataset
from transformers import DataCollatorWithPadding

from model import GATWithAdapterForLLM
from safetensors.torch import save_file

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel

)

# 兼容不同版本：prepare_model_for_kbit_training (新)，prepare_model_for_int8_training (旧)
try:
    # peft >= 0.4 的接口（更通用，支持 k-bit）
    from peft import prepare_model_for_kbit_training as prepare_model_for_quant_training

    print("from peft import prepare_model_for_kbit_training as prepare_model_for_quant_training")
except Exception:
    try:
        # 旧版本接口名
        from peft import prepare_model_for_int8_training as prepare_model_for_quant_training

        print("from peft import prepare_model_for_int8_training as prepare_model_for_quant_training")
    except Exception:
        # 如果都不可用，定义空壳（后面会判断是否需要调用）
        prepare_model_for_quant_training = None
        print("prepare_model_for_quant_training = None")

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils.prompter import Prompter


def train(
        fold: int = 0,
        base_model: str = "",  # the only required argument  --base_model /path/to/model
        data_path: str = "",
        val_data_path: str = "",
        output_dir: str = "",
        model_save_path: str = "",
        # training hyperparams
        batch_size: int = 16,
        micro_batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 512,
        val_set_size: int = 0,
        # lora hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        num_prefix: int = 1,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     torch_dtype=torch.float16,
    # )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                  user_prompt_len:
                                              ]  # could be sped up, probably
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    slama_model = GATWithAdapterForLLM(model, freeze_gnn=False, fold=fold)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_data_path.endswith(".json") or val_data_path.endswith(".jsonl"):
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None

    # 处理训练集
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    # 处理验证集（如果 val_data_path 存在）
    if val_data_path:
        val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt)
    else:
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=slama_model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data is not None else None,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="no",
            # evaluation_strategy="steps",
            # eval_steps=2500,
            save_steps=None,
            save_total_limit=None,
            ddp_find_unused_parameters=False if ddp else None,
            # load_best_model_at_end=True if val_set_size > 0 else False,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to=None,
            run_name=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    slama_model.llm.save_pretrained(output_dir, safe_serialization=True)
    save_path = model_save_path

    state_to_save = {
        "gat": {k: v.cpu() for k, v in slama_model.gat.state_dict().items()},
        "gcn": {k: v.cpu() for k, v in slama_model.gcn.state_dict().items()},
        "node2prefix": {k: v.cpu() for k, v in slama_model.node2prefix.state_dict().items()},
        "alpha": slama_model.alpha.detach().cpu(),  # 融合权重 α 也保存
        "proj": {k: v.cpu() for k, v in slama_model.proj.state_dict().items()},
        "sage": {k: v.cpu() for k, v in slama_model.sage.state_dict().items()},
        "lnc_adapter": {k: v.cpu() for k, v in slama_model.lnc_adapter.state_dict().items()},
        "drug_adapter": {k: v.cpu() for k, v in slama_model.drug_adapter.state_dict().items()},
        "diff_adapter": {k: v.cpu() for k, v in slama_model.diff_adapter.state_dict().items()},
    }

    torch.save(state_to_save, save_path)
    print(f"prefix model saved to {save_path}")

    # model.save_pretrained(output_dir)
    # print(
    #     "\n If there's a warning about missing keys above, please disregard :)"
    # )

# if __name__ == "__main__":
#     fire.Fire(train)
