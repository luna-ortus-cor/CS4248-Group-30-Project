import os
import json
import torch
import glob
from PIL import Image
from tqdm import tqdm
from typing import List
from pydantic import BaseModel

# Unsloth & Qwen utilities
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACEBOOK_DATA_DIR = os.path.join(BASE_DIR, "..", "facebook-data")
DATASET_JSONL = os.path.join(FACEBOOK_DATA_DIR, "train.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "train")
CHUNK_SIZE = 500   # records per output file
BATCH_SIZE = 8


# --- STRUCTURED OUTPUT SCHEMA ---
class Metaphor(BaseModel):
    metaphor: str
    meaning: str

class MemeAnalysis(BaseModel):
    img_captions: List[str]
    meme_captions: List[str]
    title: str
    metaphors: List[Metaphor]


# --- RESUME: collect already-captioned post_ids ---
def load_already_captioned_ids():
    seen = set()
    # Check existing captions_output*.jsonl in parent dir
    for path in glob.glob(os.path.join(BASE_DIR, "captions_output*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen.add(str(d["post_id"]))
                except Exception:
                    pass
    # Check any already-generated files in train/
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen.add(str(d["post_id"]))
                except Exception:
                    pass
    return seen


# --- LOAD ALL TRAINING RECORDS, EXCLUDING ALREADY CAPTIONED ---
def load_remaining_records(already_captioned):
    records = []
    with open(DATASET_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if str(d["id"]) not in already_captioned:
                records.append(d)
    return records


# --- DETERMINE NEXT OUTPUT FILE INDEX ---
def next_chunk_index():
    existing = glob.glob(os.path.join(OUTPUT_DIR, "captions_train_chunk*.jsonl"))
    if not existing:
        return 1
    indices = []
    for p in existing:
        name = os.path.basename(p)
        try:
            idx = int(name.replace("captions_train_chunk", "").replace(".jsonl", ""))
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1


# --- MAIN ---
def process_memes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    already_captioned = load_already_captioned_ids()
    print(f"Already captioned: {len(already_captioned)} records")

    remaining = load_remaining_records(already_captioned)
    print(f"Remaining to caption: {len(remaining)} records")

    if not remaining:
        print("All training data already captioned!")
        return

    print(f"Loading model: {MODEL_ID}")
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)

    prompt_text = (
        "Analyze this meme and extract its components. "
        "1. Provide all literal 'img_captions' describing exactly what is visually seen. "
        "2. Provide 1 'meme_captions' explaining the underlying joke or message. "
        "3. Give the meme a short 'title'. "
        "4. Identify 'metaphors' where a visual element represents a broader meaning. "
        "You MUST output exactly in valid JSON format matching this schema:\n"
        f"{MemeAnalysis.model_json_schema()}"
    )

    chunk_idx = next_chunk_index()
    chunk_records = []  # buffer for current chunk
    total_written = 0

    def flush_chunk(records, idx):
        out_path = os.path.join(OUTPUT_DIR, f"captions_train_chunk{idx:03d}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nSaved chunk {idx} -> {out_path} ({len(records)} records)")

    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Captioning batches"):
        batch_data = remaining[i : i + BATCH_SIZE]
        batch_messages = []

        for data in batch_data:
            img_path = os.path.join(FACEBOOK_DATA_DIR, data["img"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            batch_messages.append(messages)

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            do_sample=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for data, raw_output in zip(batch_data, output_texts):
            try:
                clean_json = raw_output.strip().replace("```json", "").replace("```", "")
                result_json = json.loads(clean_json)
                final_output = {
                    "category": "HMD-memes",
                    "img_captions": result_json.get("img_captions", []),
                    "meme_captions": result_json.get("meme_captions", []),
                    "title": result_json.get("title", ""),
                    "img_fname": data["img"],
                    "metaphors": result_json.get("metaphors", []),
                    "post_id": str(data["id"]),
                    "label": int(data["label"]),
                    "text": data.get("text", ""),
                    "raw_output": raw_output,
                }
            except Exception as e:
                print(f"\nError parsing JSON for ID {data['id']}: {e}\nRaw: {raw_output[:200]}")
                final_output = {
                    "category": "HMD-memes",
                    "img_captions": [],
                    "meme_captions": [],
                    "title": "",
                    "img_fname": data["img"],
                    "metaphors": [],
                    "post_id": str(data["id"]),
                    "label": int(data["label"]),
                    "text": data.get("text", ""),
                    "raw_output": raw_output,
                }

            chunk_records.append(final_output)
            total_written += 1

            # Flush chunk when full
            if len(chunk_records) >= CHUNK_SIZE:
                flush_chunk(chunk_records, chunk_idx)
                chunk_idx += 1
                chunk_records = []

    # Flush any remaining records
    if chunk_records:
        flush_chunk(chunk_records, chunk_idx)

    print(f"\nDone. Total records written: {total_written}")


if __name__ == "__main__":
    process_memes()
