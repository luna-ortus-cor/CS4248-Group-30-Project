"""
Zero-shot inference on dev.jsonl using the base Qwen3-VL-8B (no fine-tuning).
Uses the same prompt format as finetune_and_infer.py for fair comparison.

Usage:
  python sft/zeroshot_infer.py
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
FACEBOOK_DIR = BASE_DIR / "facebook-data"
DEV_JSONL    = FACEBOOK_DIR / "dev.jsonl"
OUTPUT_DIR   = BASE_DIR / "sft" / "output"
OUTPUT_FILE  = OUTPUT_DIR / "zeroshot_dev_predictions.jsonl"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
BATCH_SIZE  = 8
MAX_NEW_TOKENS = 128

CAPTION_PATTERNS = [
    BASE_DIR / "hateful-captioning" / "train" / "captions_train_chunk*.jsonl",
    BASE_DIR / "captioned-hmd-data" / "captions_output*.jsonl",
    BASE_DIR / "hateful-captioning" / "captions_output*.jsonl",
]

SYSTEM_PROMPT = (
    "You are an expert at detecting hateful content in internet memes. "
    "You will be given a meme image along with its text overlay and an analysis "
    "of the meme's visual content and meaning. "
    "Classify the meme as hateful (1) or not hateful (0). "
    "Respond with a brief reasoning followed by your final answer on the last line "
    "as exactly: LABEL: 0  or  LABEL: 1"
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_captions() -> dict:
    captions = {}
    for pattern in CAPTION_PATTERNS:
        for path in sorted(glob.glob(str(pattern))):
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        pid = str(d["post_id"])
                        if pid not in captions and d.get("img_captions"):
                            captions[pid] = d
                    except Exception:
                        pass
    print(f"Loaded captions for {len(captions)} post_ids")
    return captions


def build_caption_context(cap: Optional[dict]) -> str:
    if not cap:
        return ""
    parts = []
    if cap.get("title"):
        parts.append(f"Title: {cap['title']}")
    if cap.get("img_captions"):
        parts.append("Visual description: " + " ".join(cap["img_captions"]))
    if cap.get("meme_captions"):
        parts.append("Meme meaning: " + " ".join(cap["meme_captions"]))
    if cap.get("metaphors"):
        metaphor_strs = [
            f"{m['metaphor']} → {m['meaning']}"
            for m in cap["metaphors"]
            if isinstance(m, dict) and m.get("metaphor")
        ]
        if metaphor_strs:
            parts.append("Metaphors: " + "; ".join(metaphor_strs))
    return "\n".join(parts)


def load_dev_records() -> list:
    records = []
    with open(DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            img_path = str(FACEBOOK_DIR / d["img"])
            if not os.path.exists(img_path):
                continue
            records.append({
                "post_id":  str(d["id"]),
                "img_path": img_path,
                "text":     d.get("text", ""),
                "label":    int(d["label"]),
            })
    print(f"Loaded {len(records)} dev records")
    return records


def parse_label(text: str) -> int:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("LABEL:"):
            val = line.split(":", 1)[1].strip()
            if val in ("0", "1"):
                return int(val)
    for ch in reversed(text):
        if ch in ("0", "1"):
            return int(ch)
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    captions = load_captions()
    records  = load_dev_records()

    print(f"\nLoading base model: {MODEL_ID}")
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with open(OUTPUT_FILE, "w") as f_out:
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Zero-shot inference"):
            batch = records[i : i + BATCH_SIZE]
            batch_messages = []

            for rec in batch:
                cap         = captions.get(rec["post_id"])
                cap_context = build_caption_context(cap)
                user_text   = f'Meme text overlay: "{rec["text"]}"'
                if cap_context:
                    user_text += f"\n\nMeme analysis:\n{cap_context}"
                user_text += "\n\nIs this meme hateful? Classify as 0 (not hateful) or 1 (hateful)."

                batch_messages.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{rec['img_path']}"},
                        {"type": "text",  "text": user_text},
                    ]},
                ])

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=texts, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
            )
            generated_ids_trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for rec, raw in zip(batch, output_texts):
                pred = parse_label(raw)
                row  = {
                    "id":         rec["post_id"],
                    "label":      pred,
                    "true_label": rec["label"],
                    "raw_output": raw,
                }
                f_out.write(json.dumps(row) + "\n")
                results.append(row)

            f_out.flush()

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = [r["true_label"] for r in results]
    y_pred = [r["label"]      for r in results]

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    print(f"\nPredictions saved to {OUTPUT_FILE}")
    print(f"\n{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUROC    : {auroc:.4f}")
    print(f"{'='*40}")
    counts = {0: y_pred.count(0), 1: y_pred.count(1)}
    print(f"Predicted: {counts[0]} non-hateful, {counts[1]} hateful")


if __name__ == "__main__":
    main()