from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

from PIL import Image

from .io import stable_id, write_jsonl


ANATOMY_TERMS = (
    "abdomen", "brain", "breast", "chest", "colon", "eye", "femur", "heart",
    "hip", "kidney", "knee", "liver", "lung", "neck", "pelvis", "prostate",
    "rib", "skull", "spine", "stomach", "thorax", "uterus",
)


def infer_organ(question: str) -> str:
    text = question.lower()
    for term in ANATOMY_TERMS:
        if re.search(rf"\b{re.escape(term)}s?\b", text):
            return term
    return "radiology anatomy"


def _save_split(dataset, split: str, limit: int | None, root: Path) -> list[dict]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, item in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        image: Image.Image = item["image"].convert("RGB")
        source_name = str(item.get("image_name") or item.get("image_id") or index)
        # Hash pixels because some dataset conversions omit the original image ID.
        # Repeated QA rows then share one augmentation and cannot cross a split.
        image_id = hashlib.sha256(
            f"{image.width}x{image.height}:".encode() + image.tobytes()
        ).hexdigest()[:16]
        image_path = images_dir / f"{image_id}.png"
        if not image_path.exists():
            image.save(image_path)
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()
        rows.append(
            {
                "id": stable_id(f"{split}:{source_name}:{question}:{answer}"),
                "image_id": image_id,
                "image": str(image_path.relative_to(root)),
                "question": question,
                "answer": answer,
                "split": split,
                "organ": str(item.get("image_organ") or infer_organ(question)).lower(),
                "answer_type": str(item.get("answer_type") or "unknown").lower(),
                "question_type": str(
                    item.get("question_type_primary") or item.get("question_type") or "unknown"
                ).lower(),
            }
        )
    return rows


def image_grouped_resplit(rows: list[dict], test_fraction: float, seed: int) -> None:
    image_ids = sorted({row["image_id"] for row in rows})
    random.Random(seed).shuffle(image_ids)
    cut = max(1, round(len(image_ids) * test_fraction))
    test_ids = set(image_ids[:cut])
    for row in rows:
        row["split"] = "test" if row["image_id"] in test_ids else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a compact VQA-RAD manifest")
    parser.add_argument("--output", type=Path, default=Path("data/vqarad"))
    parser.add_argument("--dataset", default="abhay2812/vqa-rad")
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-test", type=int, default=96)
    parser.add_argument(
        "--keep-published-split",
        action="store_true",
        help="Keep QA splits; default is the paper's image-grouped 70:30 split",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset(args.dataset)
    rows = _save_split(dataset["train"], "train", args.max_train, args.output)
    rows += _save_split(dataset["test"], "test", args.max_test, args.output)
    if not args.keep_published_split:
        image_grouped_resplit(rows, 0.30, args.seed)
    write_jsonl(args.output / "manifest.jsonl", rows)
    print(f"Wrote {len(rows)} QA rows to {args.output / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
