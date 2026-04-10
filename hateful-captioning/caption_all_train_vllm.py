import os
import json
import glob
from PIL import Image
from tqdm import tqdm
from typing import List
from pydantic import BaseModel

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoProcessor

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACEBOOK_DATA_DIR = os.path.join(BASE_DIR, "..", "facebook-data")
DATASET_JSONL = os.path.join(FACEBOOK_DATA_DIR, "train.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "train")
CHUNK_SIZE = 500
BATCH_SIZE = 64   # vLLM handles continuous batching — much larger batches are fine


# --- STRUCTURED OUTPUT SCHEMA ---
class Metaphor(BaseModel):
    metaphor: str
    meaning: str

class MemeAnalysis(BaseModel):
    img_captions: List[str]
    meme_captions: List[str]
    title: str
    metaphors: List[Metaphor]


PROMPT_TEXT = (
    "Analyze this meme and extract its components. "
    "1. Provide all literal 'img_captions' describing exactly what is visually seen. "
    "2. Provide 1 'meme_captions' explaining the underlying joke or message. "
    "3. Give the meme a short 'title'. "
    "4. Identify 'metaphors' where a visual element represents a broader meaning. "
    "You MUST output exactly in valid JSON format matching this schema:\n"
    f"{MemeAnalysis.model_json_schema()}"
)


# --- RESUME: collect already-captioned post_ids ---
def load_already_captioned_ids():
    seen = set()
    for path in glob.glob(os.path.join(BASE_DIR, "captions_output*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    seen.add(str(json.loads(line)["post_id"]))
                except Exception:
                    pass
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    seen.add(str(json.loads(line)["post_id"]))
                except Exception:
                    pass
    return seen


def load_remaining_records(already_captioned):
    records = []
    with open(DATASET_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if str(d["id"]) not in already_captioned:
                records.append(d)
    return records


def next_chunk_index():
    existing = glob.glob(os.path.join(OUTPUT_DIR, "captions_train_chunk*.jsonl"))
    if not existing:
        return 1
    indices = []
    for p in existing:
        try:
            idx = int(os.path.basename(p).replace("captions_train_chunk", "").replace(".jsonl", ""))
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1


def flush_chunk(records, idx):
    out_path = os.path.join(OUTPUT_DIR, f"captions_train_chunk{idx:03d}.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved chunk {idx} -> {out_path} ({len(records)} records)")


def build_prompt(processor, img_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_output(raw_output):
    clean = raw_output.strip().replace("```json", "").replace("```", "")
    # Strip vLLM thinking tags if present
    if "</think>" in clean:
        clean = clean.split("</think>", 1)[-1].strip()
    return json.loads(clean)


def process_memes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    already_captioned = load_already_captioned_ids()
    print(f"Already captioned: {len(already_captioned)} records")

    remaining = load_remaining_records(already_captioned)
    print(f"Remaining to caption: {len(remaining)} records")

    if not remaining:
        print("All training data already captioned!")
        return

    print(f"Loading vLLM model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=BATCH_SIZE,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.90,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=1024,
    )

    chunk_idx = next_chunk_index()
    chunk_records = []
    total_written = 0

    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Captioning batches"):
        batch_data = remaining[i : i + BATCH_SIZE]

        # Build vLLM multimodal inputs
        inputs = []
        for data in batch_data:
            img_path = os.path.join(FACEBOOK_DATA_DIR, data["img"])
            prompt = build_prompt(processor, img_path)
            image = Image.open(img_path).convert("RGB")
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            })

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for data, output in zip(batch_data, outputs):
            raw_output = output.outputs[0].text
            try:
                result_json = parse_output(raw_output)
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

            if len(chunk_records) >= CHUNK_SIZE:
                flush_chunk(chunk_records, chunk_idx)
                chunk_idx += 1
                chunk_records = []

    if chunk_records:
        flush_chunk(chunk_records, chunk_idx)

    print(f"\nDone. Total records written: {total_written}")


if __name__ == "__main__":
    process_memes()
