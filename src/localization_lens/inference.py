from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from .io import read_jsonl


def normalize_answer(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def token_recall(prediction: str, reference: str) -> float:
    predicted = set(normalize_answer(prediction).split())
    expected = set(normalize_answer(reference).split())
    if not expected:
        return float(not predicted)
    return len(predicted & expected) / len(expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original-image-only VQA inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_path = Path(args.model)
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.is_file():
        from peft import PeftModel

        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base = Idefics3ForConditionalGeneration.from_pretrained(
            config["base_model_name_or_path"], torch_dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=dtype, device_map="auto"
        )
    if args.image and args.question:
        samples = [{"id": "single", "image": str(args.image), "question": args.question}]
        root = Path(".")
    elif args.manifest:
        root = args.manifest.parent
        samples = [row for row in read_jsonl(args.manifest) if row["split"] == args.split][: args.max_samples]
    else:
        raise SystemExit("Provide --image and --question, or --manifest")
    predictions = []
    correct = 0
    closed_correct = closed_total = 0
    open_recall = []
    for row in samples:
        image = Image.open(root / row["image"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Answer briefly using the medical image."},
            {"type": "image"}, {"type": "text", "text": row["question"]},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        answer_tokens = generated[:, inputs["input_ids"].shape[1]:]
        prediction = processor.batch_decode(answer_tokens, skip_special_tokens=True)[0].strip()
        result = {"id": row["id"], "question": row["question"], "prediction": prediction}
        if "answer" in row:
            result["answer"] = row["answer"]
            result["exact_match"] = normalize_answer(prediction) == normalize_answer(row["answer"])
            correct += int(result["exact_match"])
            answer_type = row.get("answer_type", "unknown").lower()
            result["answer_type"] = answer_type
            if answer_type == "closed":
                closed_total += 1
                closed_correct += int(result["exact_match"])
            elif answer_type == "open":
                result["token_recall"] = token_recall(prediction, row["answer"])
                open_recall.append(result["token_recall"])
        predictions.append(result)
        print(json.dumps(result, ensure_ascii=False))
    with args.output.open("w", encoding="utf-8") as stream:
        for result in predictions:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    if predictions and "answer" in predictions[0]:
        print(f"Exact match: {correct / len(predictions):.3f} ({correct}/{len(predictions)})")
        if closed_total:
            print(f"Closed accuracy: {closed_correct / closed_total:.3f} ({closed_correct}/{closed_total})")
        if open_recall:
            print(f"Open token recall: {sum(open_recall) / len(open_recall):.3f}")


if __name__ == "__main__":
    main()
