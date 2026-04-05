"""
STEP 2 — Baseline Inference: VLM (4-bit) + SigLIP RAG
======================================================
Input:  facebook-data/dev.jsonl
        facebook-data/img/
        pipeline/rag_store/embeddings.npy  (or none for zero-shot)

Output: output/preds_<model_slug>_4bit_<rag_variant>.jsonl
          one JSON object per line: {"id": …, "label": 0|1, …}

Usage:
  # run all models with RAG (default)
  python pipeline/run_baseline.py

  # single model, specific RAG variant
  python pipeline/run_baseline.py --model qwen3vl --rag rag
  python pipeline/run_baseline.py --model qwen3vl --rag norag

  # full ablation: all models x both variants (3 models -> 6 output files)
  python pipeline/run_baseline.py --rag all

Supported --model keys  (see MODEL_REGISTRY below):
  qwen3vl           Qwen/Qwen3-VL-8B-Instruct
  qwen3vl-thinking  Qwen/Qwen3-VL-8B-Thinking
  llama4-scout      meta-llama/Llama-4-Scout-17B-16E-Instruct   (gated)
  llama3-vision     meta-llama/Llama-3.2-11B-Vision-Instruct    (gated)
  llava             llava-hf/llava-v1.6-mistral-7b-hf

Supported --rag keys:
  rag    use SigLIP RAG retrieval (retrieved few-shot examples in prompt)
  norag  zero-shot, no retrieved context
  all    run both variants sequentially
"""

import argparse
import json
import pickle
import re
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    SiglipModel,
    SiglipProcessor,
)

# -- Paths ---------------------------------------------------------------------
BASE       = Path(__file__).resolve().parents[1]
TEST_JSONL = BASE / "facebook-data" / "dev.jsonl"
FB_IMG_DIR = BASE / "facebook-data" / "img"
RAG_STORES = {
    "memecap": BASE / "pipeline" / "rag_store_memecap",
    "full":    BASE / "pipeline" / "rag_store_full",
}
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGLIP_MODEL    = "google/siglip-base-patch16-224"
RAG_K           = 3
DEFAULT_RAG_STORE = "full"

# RAG variants: slug suffix -> whether to use retrieval
RAG_VARIANTS = {
    "rag":   True,   # use SigLIP RAG (few-shot retrieved examples)
    "norag": False,  # zero-shot, no context
}
DEFAULT_RAG = "rag"


# -- Model registry ------------------------------------------------------------
MODEL_REGISTRY = {
    "qwen3vl": {
        "hf_id":          "Qwen/Qwen3-VL-8B-Instruct",
        "slug":           "qwen3vl_8b",
        "loader":         "auto",
        "max_new_tokens": 64,
    },
    "llava": {
        "hf_id":          "llava-hf/llava-v1.6-mistral-7b-hf",
        "slug":           "llava_v16_mistral_7b",
        "loader":         "llava",
        "max_new_tokens": 64,
    },
}


# -- SigLIP retriever ----------------------------------------------------------

class SigLIPRetriever:
    def __init__(self, embed_path: Path, meta_path: Path, device: str):
        self.device = device
        self.embeddings = np.load(embed_path).astype("float32")
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.where(norms == 0, 1, norms)
        print(f"Loaded {len(self.embeddings)} RAG embeddings  (dim={self.embeddings.shape[1]})")
        print(f"Loading SigLIP ({SIGLIP_MODEL}) for query encoding...")
        self.model     = SiglipModel.from_pretrained(SIGLIP_MODEL).to(device).eval()
        self.processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL)

    def retrieve(self, image: Image.Image, text: str, k: int = RAG_K) -> list:
        inputs = self.processor(
            text=[text or " "], images=[image],
            return_tensors="pt", padding="max_length", truncation=True,
        ).to(self.device)
        with torch.no_grad():
            out        = self.model(**inputs)
            img_embed  = out.image_embeds[0].cpu().float().numpy()
            text_embed = out.text_embeds[0].cpu().float().numpy()
        img_embed  /= np.linalg.norm(img_embed)  + 1e-8
        text_embed /= np.linalg.norm(text_embed) + 1e-8
        query = np.concatenate([img_embed, text_embed])
        query /= np.linalg.norm(query) + 1e-8
        scores = self.embeddings @ query
        top_k  = np.argsort(scores)[-k:][::-1]
        contexts = []
        for idx in top_k:
            m = self.metadata[idx]
            lines = []
            if m.get("title"):         lines.append(f"Title: {m['title']}")
            if m.get("meme_captions"): lines.append(f"Meaning: {m['meme_captions'][0]}")
            if m.get("img_captions"):  lines.append(f"Visual: {m['img_captions'][0]}")
            contexts.append("\n".join(lines))
        return contexts


# -- Prompt & output parsing ---------------------------------------------------

def build_prompt(meme_text: str, rag_contexts: list) -> str:
    """
    If rag_contexts is non-empty, inject them as retrieved references to help decode
    figurative meaning and cultural context.
    If empty (no-RAG / zero-shot), the context block is omitted entirely.
    """
    prompt = (
        "You are a hate speech detection expert. "
        "A meme is hateful if it attacks or demeans a person or group based on a protected characteristic "
        "(race, religion, gender, nationality, sexual orientation, disability). "
        "Harmful stereotypes, dehumanisation, and slurs all count — even when framed as jokes.\n\n"
    )
    if rag_contexts:
        context_block = "\n\n".join(
            f"[Reference {i+1}]\n{ctx}" for i, ctx in enumerate(rag_contexts)
        )
        prompt += (
            "The following similar memes may help you interpret the figurative meaning and cultural context:\n"
            f"{context_block}\n\n"
        )
    prompt += (
        f'Meme text: "{meme_text}"\n\n'
        "Does this meme target a protected group or promote hatred? "
        "Respond with ONLY valid JSON — no explanation, no markdown:\n"
        '{"label": 0}  ->  not hateful\n'
        '{"label": 1}  ->  hateful'
    )
    return prompt


def extract_thinking(raw: str) -> tuple:
    match = re.search(r'<think>(.*?)</think>(.*)', raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", raw.strip()


def parse_label(answer: str) -> int:
    match = re.search(r'"label"\s*:\s*([01])', answer)
    if match:
        return int(match.group(1))
    match = re.search(r'\b([01])\b', answer)
    return int(match.group(1)) if match else 0


# -- Model loaders -------------------------------------------------------------

def load_model(cfg: dict):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    if cfg["loader"] == "llava":
        model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg["hf_id"], quantization_config=bnb_cfg, device_map="auto",
        )
        processor = LlavaNextProcessor.from_pretrained(cfg["hf_id"])
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["hf_id"], quantization_config=bnb_cfg, device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(cfg["hf_id"])
    return model, processor


def run_inference(model, processor, image, prompt_text: str, cfg: dict) -> str:
    if cfg["loader"] == "llava":
        prompt = f"[INST] <image>\n{prompt_text} [/INST]"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    else:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  prompt_text},
        ]}]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
    input_len = inputs.input_ids.shape[1]
    return processor.decode(gen_ids[0][input_len:], skip_special_tokens=True)


# -- Per-model inference loop --------------------------------------------------

def run_model(cfg: dict, samples: list, retriever, rag_variant: str) -> Path:
    """
    Run inference for one model + one RAG variant.
    retriever is SigLIPRetriever or None (zero-shot).
    Output: output/preds_<slug>_4bit_<rag_variant>.jsonl
    """
    output_file = OUTPUT_DIR / f"preds_{cfg['slug']}_4bit_{rag_variant}.jsonl"

    print(f"\nLoading {cfg['hf_id']} with 4-bit quantization...")
    model, processor = load_model(cfg)
    print("Model loaded.")

    already_done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming - {len(already_done)} samples already processed.")

    with open(output_file, "a") as out_f:
        for sample in tqdm(samples, desc=f"{cfg['slug']} [{rag_variant}]", file=sys.stdout):
            sample_id = sample["id"]
            if sample_id in already_done:
                continue

            img_name  = Path(sample.get("img", "")).name
            img_path  = FB_IMG_DIR / img_name
            meme_text = sample.get("text", "")

            try:
                image = Image.open(img_path).convert("RGB")
                image.thumbnail((512, 512))
            except Exception as e:
                out_f.write(json.dumps({"id": sample_id, "label": 0, "raw_output": "", "error": str(e)}) + "\n")
                out_f.flush()
                continue

            # RAG retrieval -- skipped when retriever is None (zero-shot)
            contexts = []
            if retriever is not None:
                try:
                    contexts = retriever.retrieve(image, meme_text, k=RAG_K)
                except Exception:
                    contexts = []

            prompt_text = build_prompt(meme_text, contexts)

            try:
                raw_output = run_inference(model, processor, image, prompt_text, cfg)
            except Exception as e:
                out_f.write(json.dumps({"id": sample_id, "label": 0, "raw_output": "", "error": str(e)}) + "\n")
                out_f.flush()
                continue

            thinking_text, answer_text = extract_thinking(raw_output)
            label = parse_label(answer_text)
            out_f.write(json.dumps({
                "id":         sample_id,
                "label":      label,
                "thinking":   thinking_text,
                "answer":     answer_text,
                "raw_output": raw_output,
            }) + "\n")
            out_f.flush()
            sys.stdout.flush()

    print(f"Predictions saved -> {output_file}")
    del model
    torch.cuda.empty_cache()
    return output_file


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline VLM inference - RAG vs no-RAG ablation")
    parser.add_argument(
        "--model", default=None, choices=list(MODEL_REGISTRY.keys()),
        help="Single model to run. Omit to run all models.",
    )
    parser.add_argument(
        "--rag", default=DEFAULT_RAG,
        choices=list(RAG_VARIANTS.keys()) + ["all"],
        help=(
            "'rag' = SigLIP retrieved few-shot context (default), "
            "'norag' = zero-shot no context, "
            "'all' = run both variants (doubles output files)"
        ),
    )
    parser.add_argument(
        "--rag-store", default=DEFAULT_RAG_STORE,
        choices=list(RAG_STORES.keys()),
        help=(
            "'memecap' = MemeCap-only RAG store (default), "
            "'full' = MemeCap + captioned HMD store"
        ),
    )
    args = parser.parse_args()

    models_to_run   = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    variants_to_run = list(RAG_VARIANTS.keys()) if args.rag == "all" else [args.rag]
    total           = len(models_to_run) * len(variants_to_run)
    rag_store_dir   = RAG_STORES[args.rag_store]
    embed_path      = rag_store_dir / "embeddings.npy"
    meta_path       = rag_store_dir / "metadata.pkl"

    print("=" * 60)
    print("Step 2 -- Baseline Inference")
    print(f"Models  : {', '.join(models_to_run)}")
    print(f"RAG     : {', '.join(variants_to_run)}")
    print(f"Store   : {args.rag_store}  ({rag_store_dir})")
    print(f"Total   : {total} runs -> {total} output files")
    print("=" * 60)

    with open(TEST_JSONL) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(samples)} samples from {TEST_JSONL.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    need_rag = any(RAG_VARIANTS[v] for v in variants_to_run)
    if need_rag and not embed_path.exists():
        print(f"ERROR: RAG store not found at {rag_store_dir}")
        print(f"  Run: python pipeline/build_siglip_rag.py --store {args.rag_store}")
        raise SystemExit(1)

    failed = []
    run_n  = 0
    for model_key in models_to_run:
        cfg = MODEL_REGISTRY[model_key]
        for variant in variants_to_run:
            run_n += 1
            use_rag = RAG_VARIANTS[variant]
            print(f"\n{'=' * 60}")
            print(f"Run {run_n}/{total}: {cfg['hf_id']}  [{'RAG' if use_rag else 'no RAG'}]")
            print(f"{'=' * 60}")

            # Load SigLIP only when this variant needs it; free it before loading VLM
            retriever = None
            if use_rag:
                print("Loading SigLIP RAG store...")
                retriever = SigLIPRetriever(embed_path, meta_path, device)

            try:
                run_model(cfg, samples, retriever, variant)
            except Exception as e:
                print(f"ERROR: {model_key} [{variant}]: {e}")
                failed.append(f"{model_key}+{variant}")
            finally:
                # Free SigLIP VRAM before next VLM load
                if retriever is not None:
                    del retriever.model
                    del retriever
                    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("All runs complete.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("Run pipeline/evaluate.py to compute metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
