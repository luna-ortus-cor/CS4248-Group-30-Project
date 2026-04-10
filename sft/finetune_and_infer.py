"""
Fine-tune Qwen3-VL-8B-Instruct (4-bit LoRA) on the Facebook HMD training set,
then run inference on the HMD test set.

Usage:
  python sft/finetune_and_infer.py              # train + infer
  python sft/finetune_and_infer.py --infer-only # skip training, load saved adapter
"""

import os
import json
import glob
import argparse
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]
FACEBOOK_DIR    = BASE_DIR / "facebook-data"
TRAIN_JSONL     = FACEBOOK_DIR / "train.jsonl"
TEST_JSONL      = FACEBOOK_DIR / "test.jsonl"
DEV_JSONL       = FACEBOOK_DIR / "dev.jsonl"
CAPTION_DIRS    = [
    BASE_DIR / "hateful-captioning" / "train",        # 8500 new captions
    BASE_DIR / "captioned-hmd-data",                  # original 1197 (if present)
    BASE_DIR / "hateful-captioning",                  # captions_output*.jsonl
]
OUTPUT_DIR      = BASE_DIR / "sft" / "output"
ADAPTER_DIR     = OUTPUT_DIR / "qwen3vl_hmd_lora"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.jsonl"
DEV_PREDICTIONS_FILE = OUTPUT_DIR / "dev_predictions.jsonl"

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_ID        = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
LORA_RANK       = 16
LORA_ALPHA      = 16
MAX_SEQ_LEN     = 2048
TRAIN_EPOCHS    = 3
BATCH_SIZE      = 4
GRAD_ACCUM      = 4          # effective batch = 16
LR              = 2e-4
INFER_BATCH     = 8
MAX_NEW_TOKENS  = 128


# ── Prompt helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert at detecting hateful content in internet memes. "
    "You will be given a meme image along with its text overlay and an analysis "
    "of the meme's visual content and meaning. "
    "Classify the meme as hateful (1) or not hateful (0). "
    "Respond with a brief reasoning followed by your final answer on the last line "
    "as exactly: LABEL: 0  or  LABEL: 1"
)


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


def build_user_message(img_path: str, meme_text: str, caption_context: str) -> list:
    content = [{"type": "image", "image": f"file://{img_path}"}]
    user_text = f'Meme text overlay: "{meme_text}"'
    if caption_context:
        user_text += f"\n\nMeme analysis:\n{caption_context}"
    user_text += "\n\nIs this meme hateful? Classify as 0 (not hateful) or 1 (hateful)."
    content.append({"type": "text", "text": user_text})
    return content


def label_to_response(label: int) -> str:
    if label == 1:
        return "This meme targets individuals or groups based on protected characteristics, promoting hatred or dehumanization.\nLABEL: 1"
    else:
        return "This meme does not promote hatred or target protected characteristics.\nLABEL: 0"


def parse_label(text: str) -> int:
    """Extract predicted label from model output."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("LABEL:"):
            val = line.split(":", 1)[1].strip()
            if val in ("0", "1"):
                return int(val)
    # fallback: scan for last occurrence of 0 or 1
    for ch in reversed(text):
        if ch in ("0", "1"):
            return int(ch)
    return 0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_captions() -> dict:
    """Load all captions keyed by post_id."""
    captions = {}
    patterns = [
        BASE_DIR / "hateful-captioning" / "train" / "captions_train_chunk*.jsonl",
        BASE_DIR / "captioned-hmd-data" / "captions_output*.jsonl",
        BASE_DIR / "hateful-captioning" / "captions_output*.jsonl",
    ]
    for pattern in patterns:
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
    print(f"Loaded captions for {len(captions)} unique post_ids")
    return captions


def load_train_records(captions: dict) -> list:
    records = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            pid = str(d["id"])
            img_path = str(FACEBOOK_DIR / d["img"])
            if not os.path.exists(img_path):
                continue
            records.append({
                "post_id": pid,
                "img_path": img_path,
                "text": d.get("text", ""),
                "label": int(d["label"]),
                "caption": captions.get(pid),
            })
    print(f"Loaded {len(records)} training records")
    return records


def load_test_records() -> list:
    records = []
    with open(TEST_JSONL) as f:
        for line in f:
            d = json.loads(line)
            img_path = str(FACEBOOK_DIR / d["img"])
            if not os.path.exists(img_path):
                continue
            records.append({
                "post_id": str(d["id"]),
                "img_path": img_path,
                "text": d.get("text", ""),
            })
    print(f"Loaded {len(records)} test records")
    return records


def load_dev_records() -> list:
    records = []
    with open(DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            img_path = str(FACEBOOK_DIR / d["img"])
            if not os.path.exists(img_path):
                continue
            records.append({
                "post_id": str(d["id"]),
                "img_path": img_path,
                "text": d.get("text", ""),
                "label": int(d["label"]),
            })
    print(f"Loaded {len(records)} dev records")
    return records


# ── Training dataset ──────────────────────────────────────────────────────────

def make_conversation(record: dict) -> dict:
    """Return raw messages list for a training record (not yet serialized)."""
    cap_context = build_caption_context(record["caption"])
    user_content = build_user_message(record["img_path"], record["text"], cap_context)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
        {"role": "assistant", "content": label_to_response(record["label"])},
    ]


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def train(train_records: list):
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from qwen_vl_utils import process_vision_info

    print(f"\nLoading model: {MODEL_ID}")
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",  # 30% less VRAM, enables larger batches
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r           = LORA_RANK,
        lora_alpha  = LORA_ALPHA,
        lora_dropout= 0,
        bias        = "none",
        use_rslora  = False,
        random_state= 42,
    )

    print("Building training dataset...")
    conversations = [{"messages": make_conversation(r)} for r in train_records]

    # Use a simple wrapper so we don't serialize multimodal content through PyArrow
    class ConversationDataset(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    dataset = ConversationDataset(conversations)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = processor.tokenizer,
        data_collator = UnslothVisionDataCollator(model, processor),
        train_dataset = dataset,
        args = SFTConfig(
            per_device_train_batch_size    = BATCH_SIZE,
            gradient_accumulation_steps    = GRAD_ACCUM,
            warmup_steps                   = 50,
            num_train_epochs               = TRAIN_EPOCHS,
            learning_rate                  = LR,
            fp16                           = not is_bf16_supported(),
            bf16                           = is_bf16_supported(),
            logging_steps                  = 10,
            save_strategy                  = "epoch",
            output_dir                     = str(OUTPUT_DIR / "checkpoints"),
            optim                          = "adamw_8bit",
            weight_decay                   = 0.01,
            lr_scheduler_type              = "cosine",
            seed                           = 42,
            remove_unused_columns          = False,
            dataset_text_field             = "text",
            dataset_kwargs                 = {"skip_prepare_dataset": True},
            max_seq_length                 = MAX_SEQ_LEN,
            gradient_checkpointing         = True,            # Unsloth handles this
            dataloader_num_workers         = 4,               # parallel data loading
            dataloader_pin_memory          = True,
            report_to                      = "none",
        ),
    )

    print(f"\nTraining for {TRAIN_EPOCHS} epochs...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {ADAPTER_DIR}")
    model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))

    return model, processor


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(results: list):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = [r["true_label"] for r in results]
    y_pred = [r["label"] for r in results]
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    auroc  = roc_auc_score(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUROC    : {auroc:.4f}")
    print(f"{'='*40}")
    return {"accuracy": acc, "f1": f1, "auroc": auroc}


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(test_records: list, captions: dict, model=None, processor=None,
          output_file: Path = None):
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    if model is None:
        print(f"\nLoading adapter from {ADAPTER_DIR}")
        model, processor = FastVisionModel.from_pretrained(
            str(ADAPTER_DIR),
            load_in_4bit=True,
        )

    FastVisionModel.for_inference(model)

    out_path = output_file or PREDICTIONS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(out_path, "w") as f_out:
        for i in tqdm(range(0, len(test_records), INFER_BATCH), desc="Inference"):
            batch = test_records[i : i + INFER_BATCH]
            batch_messages = []

            for rec in batch:
                cap = captions.get(rec["post_id"])
                cap_context = build_caption_context(cap)
                user_content = build_user_message(rec["img_path"], rec["text"], cap_context)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ]
                batch_messages.append(messages)

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
                row = {"id": rec["post_id"], "label": pred, "raw_output": raw}
                f_out.write(json.dumps(row) + "\n")
                results.append(row)

            f_out.flush()

    print(f"\nPredictions saved to {out_path}")
    counts = {0: sum(1 for r in results if r["label"] == 0),
              1: sum(1 for r in results if r["label"] == 1)}
    print(f"Predicted: {counts[0]} non-hateful, {counts[1]} hateful")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-only", action="store_true",
                        help="Skip training, load saved adapter and run inference")
    parser.add_argument("--train-only", action="store_true",
                        help="Run training only, skip inference")
    parser.add_argument("--dev", action="store_true",
                        help="Run inference on dev.jsonl (has labels) and print metrics")
    args = parser.parse_args()

    captions = load_captions()

    if args.train_only:
        train_records = load_train_records(captions)
        train(train_records)
    elif args.dev:
        dev_records = load_dev_records()
        results = infer(dev_records, captions, output_file=DEV_PREDICTIONS_FILE)
        # attach true labels for evaluation
        for rec, res in zip(dev_records, results):
            res["true_label"] = rec["label"]
        evaluate(results)
    elif args.infer_only:
        test_records = load_test_records()
        infer(test_records, captions)
    else:
        train_records = load_train_records(captions)
        model, processor = train(train_records)
        test_records = load_test_records()
        infer(test_records, captions, model=model, processor=processor)


if __name__ == "__main__":
    main()
