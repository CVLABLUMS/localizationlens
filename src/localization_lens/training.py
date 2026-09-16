from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from .io import read_jsonl


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
VIEW_TYPES = {
    "unicolor": ("sam_unicolor.png",),
    "multicolor": ("sam_multicolor.png",),
    "masked": ("sam_masked.png",),
}


class LensDataset:
    def __init__(self, rows: list[dict], manifest: Path, lenses: Path, phase: str, seed: int):
        self.rows = rows
        self.root = manifest.parent
        self.lenses = lenses
        self.phase = phase
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        original = Image.open(self.root / row["image"]).convert("RGB")
        directory = self.lenses / row["image_id"]
        by_type = {
            view_type: [directory / name for name in names if (directory / name).is_file()]
            for view_type, names in VIEW_TYPES.items()
        }
        by_type = {key: paths for key, paths in by_type.items() if paths}
        if len(by_type) < 2:
            raise FileNotFoundError(
                f"Need at least two lens representation types for {row['image_id']}"
            )
        rng = random.Random(self.seed + index)
        chosen_types = rng.sample(sorted(by_type), 2)
        chosen = [rng.choice(by_type[view_type]) for view_type in chosen_types]
        if self.phase == "align":
            question = "Which anatomical region is emphasized by these localization views?"
            answer = row.get("organ") or "radiology anatomy"
        else:
            question, answer = row["question"], row["answer"]
        return {
            "images": [original, Image.open(chosen[0]).convert("RGB"), Image.open(chosen[1]).convert("RGB")],
            "question": question,
            "answer": answer,
        }


class LensCollator:
    def __init__(self, processor):
        self.processor = processor

    def _messages(self, item: dict, answer: bool) -> list[dict]:
        content = [{"type": "text", "text": "Answer briefly using the medical image."}]
        for _ in item["images"]:
            content.append({"type": "image"})
        content.append({"type": "text", "text": item["question"]})
        messages = [{"role": "user", "content": content}]
        if answer:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]})
        return messages

    def __call__(self, examples: list[dict]):
        texts = [
            self.processor.apply_chat_template(self._messages(item, True), add_generation_prompt=False).strip()
            for item in examples
        ]
        prefixes = [
            self.processor.apply_chat_template(self._messages(item, False), add_generation_prompt=True).strip()
            for item in examples
        ]
        image_batches = [item["images"] for item in examples]
        batch = self.processor(text=texts, images=image_batches, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for index, (prefix, images) in enumerate(zip(prefixes, image_batches)):
            prefix_ids = self.processor(text=prefix, images=images, return_tensors="pt")["input_ids"]
            labels[index, : prefix_ids.shape[1]] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def build_model(model_id: str, full_finetune: bool):
    import torch
    from transformers import AutoConfig, Idefics3ForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = AutoConfig.from_pretrained(model_id)
    # SmolVLM-500M's top-level published config contains a stale Llama-3 pad ID;
    # the nested text config and tokenizer correctly use token 2.
    config.pad_token_id = config.text_config.pad_token_id
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, config=config, torch_dtype=dtype
    )
    if not full_finetune:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()
    return model


def run_stage(args, processor, model, phase: str, output: Path, epochs: float):
    from transformers import Trainer, TrainingArguments
    from .modeling import DecoupledContrastiveLoss, InfoNCELoss, pool_semantic_views

    rows = read_jsonl(args.manifest)
    def has_lenses(row: dict) -> bool:
        directory = args.lenses / row["image_id"]
        available_types = sum(
            any((directory / name).is_file() for name in names)
            for names in VIEW_TYPES.values()
        )
        return available_types >= 2

    train_rows = [row for row in rows if row["split"] == "train" and has_lenses(row)][: args.max_train]
    eval_rows = [row for row in rows if row["split"] == "test" and has_lenses(row)][: args.max_eval]
    if not train_rows or not eval_rows:
        raise RuntimeError("No train/eval rows with at least two generated lens types")
    train_data = LensDataset(train_rows, args.manifest, args.lenses, phase, args.seed)
    eval_data = LensDataset(eval_rows, args.manifest, args.lenses, phase, args.seed + 10_000)

    class AlignmentTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs, output_hidden_states=True)
            loss = outputs.loss
            contrastive_weight = args.dcl_weight if phase == "align" else args.infonce_weight
            if contrastive_weight > 0:
                core = model.get_base_model() if hasattr(model, "get_base_model") else model
                vision_owner = getattr(core, "model", core)
                if not hasattr(vision_owner, "get_image_features"):
                    raise RuntimeError("This model does not expose get_image_features")
                image_outputs = vision_owner.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                )
                features = image_outputs.pooler_output
                pooled = pool_semantic_views(features, inputs["pixel_values"], num_views=3)
                hidden = outputs.hidden_states[-1]
                answer_mask = inputs["labels"].ne(-100).unsqueeze(-1)
                text_features = (hidden * answer_mask).sum(dim=1) / answer_mask.sum(dim=1).clamp_min(1)
                if phase == "align":
                    dcl = DecoupledContrastiveLoss(args.temperature)
                    contrastive = sum(dcl(pooled[:, view], text_features) for view in range(3)) / 3
                else:
                    contrastive = InfoNCELoss(args.temperature)(pooled.mean(dim=1), text_features)
                loss = loss + contrastive_weight * contrastive
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16, fp16=args.fp16, gradient_checkpointing=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=1, report_to="none", remove_unused_columns=False,
    )
    trainer = AlignmentTrainer(
        model=model, args=training_args, data_collator=LensCollator(processor),
        train_dataset=train_data, eval_dataset=eval_data,
    )
    trainer.train()
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    return trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage Localization Lens SmolVLM training")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lenses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-eval", type=int, default=96)
    parser.add_argument("--epochs-stage1", type=float, default=1)
    parser.add_argument("--epochs-stage2", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--dcl-weight", type=float, default=0.1)
    parser.add_argument("--infonce-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-finetune", action="store_true")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = build_model(args.model, args.full_finetune)
    run_stage(args, processor, model, "align", args.output / "stage1", args.epochs_stage1)
    run_stage(args, processor, model, "vqa", args.output / "stage2", args.epochs_stage2)


if __name__ == "__main__":
    main()
