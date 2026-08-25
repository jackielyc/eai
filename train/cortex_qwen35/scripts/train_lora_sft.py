#!/usr/bin/env python3
"""LoRA SFT for Qwen3.5 on Cortex System-2 ShareGPT data (transformers 5.x compatible)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    model_name_or_path: str = ""
    dataset_path: str = ""
    output_dir: str = ""
    eval_dataset_path: str | None = None
    max_seq_length: int = 2048
    max_samples: int | None = None
    eval_max_samples: int | None = 512
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    dataloader_num_workers: int = 4
    save_total_limit: int = 3
    report_to: str = "none"
    # auto | none  — "auto" shards large MoE models across visible GPUs (single process)
    device_map: str = "none"
    # none | full_shard
    fsdp: str = "none"
    fsdp_transformer_layer_cls_to_wrap: str = ""


def load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML required for yaml configs") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def build_config(args: argparse.Namespace) -> TrainConfig:
    raw: dict[str, Any] = {}
    if args.config:
        raw.update(load_config_file(Path(args.config)))
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        raw[key] = value
    known = set(TrainConfig.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in known}
    cfg = TrainConfig(**filtered)
    if not cfg.model_name_or_path or not cfg.dataset_path or not cfg.output_dir:
        raise ValueError("model_name_or_path, dataset_path, output_dir are required")
    return cfg


class ShareGPTJsonDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_seq_length: int,
        max_samples: int | None = None,
    ) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_samples is not None:
            data = data[:max_samples]
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def _assistant_span(
        self, messages: list[dict[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        prompt = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = min(int(prompt["input_ids"].shape[-1]), int(input_ids.shape[-1]))
        labels[:prompt_len] = IGNORE_INDEX
        attention_mask = full["attention_mask"][0]
        return input_ids, attention_mask, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.data[idx]["messages"]
        input_ids, attention_mask, labels = self._assistant_span(messages)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        for f in features:
            pad_len = max_len - f["input_ids"].size(0)
            batch_input_ids.append(
                torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=self.pad_token_id)
            )
            batch_attention_mask.append(
                torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0)
            )
            batch_labels.append(
                torch.nn.functional.pad(f["labels"], (0, pad_len), value=IGNORE_INDEX)
            )
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


def infer_fsdp_wrap_class(model_name_or_path: str) -> str:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_5_moe":
        return "Qwen3_5MoeDecoderLayer"
    return "Qwen3_5DecoderLayer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_device_map = cfg.device_map == "auto"
    use_fsdp = cfg.fsdp not in {"", "none", None}

    if use_device_map and world_size > 1:
        raise ValueError("device_map=auto is single-process only; do not use torchrun")

    if torch.cuda.is_available() and not use_device_map:
        torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if use_device_map:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(cfg.model_name_or_path, **load_kwargs)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    targets = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora)
    if local_rank == 0:
        model.print_trainable_parameters()

    train_ds = ShareGPTJsonDataset(
        cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.max_samples
    )
    eval_ds = None
    if cfg.eval_dataset_path:
        eval_ds = ShareGPTJsonDataset(
            cfg.eval_dataset_path,
            tokenizer,
            cfg.max_seq_length,
            cfg.eval_max_samples,
        )

    fsdp_arg = ""
    fsdp_config = None
    if use_fsdp:
        wrap = cfg.fsdp_transformer_layer_cls_to_wrap or infer_fsdp_wrap_class(
            cfg.model_name_or_path
        )
        fsdp_arg = "full_shard auto_wrap"
        fsdp_config = {"transformer_layer_cls_to_wrap": [wrap]}

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        logging_first_step=True,
        fsdp=fsdp_arg,
        fsdp_config=fsdp_config,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if local_rank == 0:
        print(f"Saved adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
