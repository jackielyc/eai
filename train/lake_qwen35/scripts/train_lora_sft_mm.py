#!/usr/bin/env python3
"""Multimodal LoRA SFT for Qwen3.5 on ShareGPT JSON with images (share_data_lake)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Disable cuDNN before importing transformers/accelerate (they may init cuDNN on import).
torch.backends.cudnn.enabled = False

from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
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
    dataloader_num_workers: int = 0
    save_total_limit: int = 3
    report_to: str = "none"
    device_map: str = "none"
    fsdp: str = "none"
    fsdp_transformer_layer_cls_to_wrap: str = ""
    image_max_pixels: int = 262144


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


def _resolve_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        content = msg["content"]
        if isinstance(content, list):
            resolved: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "image":
                    image_ref = part.get("image") or part.get("url")
                    if not image_ref:
                        raise ValueError("Image part missing 'image' path")
                    resolved.append({"type": "image", "image": Image.open(image_ref).convert("RGB")})
                else:
                    resolved.append(part)
            messages.append({"role": msg["role"], "content": resolved})
        else:
            messages.append(msg)
    return messages


def load_sharegpt_records(path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path_obj.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records
    with path_obj.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if max_samples is not None:
        data = data[:max_samples]
    return data


class MultimodalShareGPTDataset(Dataset):
    def __init__(
        self,
        path: str,
        processor,
        max_seq_length: int,
        max_samples: int | None = None,
    ) -> None:
        self.data = load_sharegpt_records(path, max_samples)
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.data)

    def _encode(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> dict[str, torch.Tensor]:
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "truncation": True,
                "max_length": self.max_seq_length,
            },
        )
        result: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"image_grid_thw", "video_grid_thw"}:
                result[key] = value
            elif value.dim() > 1 and value.shape[0] == 1:
                result[key] = value.squeeze(0)
            else:
                result[key] = value
        return result

    def __getitem__(self, idx: int) -> dict[str, Any]:
        messages = _resolve_messages(self.data[idx]["messages"])
        full = self._encode(messages, add_generation_prompt=False)
        prompt = self._encode(messages[:-1], add_generation_prompt=True)

        input_ids = full["input_ids"]
        labels = input_ids.clone()
        prompt_len = min(int(prompt["input_ids"].shape[-1]), int(input_ids.shape[-1]))
        labels[:prompt_len] = IGNORE_INDEX

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in full:
                item[key] = full[key]
        return item


def _get_rope_model(model: torch.nn.Module) -> torch.nn.Module:
    candidates: list[torch.nn.Module] = [model]
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
        if hasattr(model.base_model, "model"):
            candidates.append(model.base_model.model)
            inner = model.base_model.model
            if hasattr(inner, "model"):
                candidates.append(inner.model)
    for candidate in candidates:
        if hasattr(candidate, "get_rope_index"):
            return candidate
    raise AttributeError("Could not locate get_rope_index on model")


@dataclass
class MultimodalCollator:
    pad_token_id: int
    model: torch.nn.Module | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].shape[-1]) for f in features)
        batch_input_ids, batch_attention_mask, batch_labels, batch_mm = [], [], [], []

        for feature in features:
            input_ids = feature["input_ids"]
            if input_ids.dim() == 2:
                input_ids = input_ids.squeeze(0)
            attention_mask = feature["attention_mask"]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.squeeze(0)
            labels = feature["labels"]
            if labels.dim() == 2:
                labels = labels.squeeze(0)

            pad_len = max_len - int(input_ids.shape[-1])
            batch_input_ids.append(
                torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            )
            batch_attention_mask.append(
                torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            )
            batch_labels.append(
                torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            )
            if "mm_token_type_ids" in feature:
                mm = feature["mm_token_type_ids"]
                if mm.dim() == 2:
                    mm = mm.squeeze(0)
                batch_mm.append(torch.nn.functional.pad(mm, (0, pad_len), value=0))

        batch: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
        if batch_mm:
            batch["mm_token_type_ids"] = torch.stack(batch_mm)

        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            if pixel_values[0].dim() == 2:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            else:
                batch["pixel_values"] = torch.cat([pv.reshape(-1, pv.shape[-1]) for pv in pixel_values], dim=0)

        if "image_grid_thw" in features[0]:
            grids = []
            for feature in features:
                grid = feature["image_grid_thw"]
                if grid.dim() == 1:
                    grid = grid.unsqueeze(0)
                grids.append(grid)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        if self.model is not None and "mm_token_type_ids" in batch:
            rope_model = _get_rope_model(self.model)
            position_ids, rope_deltas = rope_model.get_rope_index(
                input_ids=batch["input_ids"],
                mm_token_type_ids=batch["mm_token_type_ids"],
                image_grid_thw=batch.get("image_grid_thw"),
                attention_mask=batch["attention_mask"].float(),
            )
            batch["position_ids"] = position_ids
            batch["rope_deltas"] = rope_deltas

        return batch


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
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
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

    processor = AutoProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.size = {
            "longest_edge": cfg.image_max_pixels,
            "shortest_edge": 65536,
        }

    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    load_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if use_device_map:
        load_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        load_kwargs["device_map"] = {"": local_rank}

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

    train_ds = MultimodalShareGPTDataset(
        cfg.dataset_path, processor, cfg.max_seq_length, cfg.max_samples
    )
    eval_ds = None
    if cfg.eval_dataset_path:
        eval_ds = MultimodalShareGPTDataset(
            cfg.eval_dataset_path,
            processor,
            cfg.max_seq_length,
            cfg.eval_max_samples,
        )

    fsdp_arg = ""
    fsdp_config = None
    if use_fsdp:
        wrap = cfg.fsdp_transformer_layer_cls_to_wrap or infer_fsdp_wrap_class(cfg.model_name_or_path)
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
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=MultimodalCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            model=model,
        ),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    if local_rank == 0:
        print(f"Saved adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
