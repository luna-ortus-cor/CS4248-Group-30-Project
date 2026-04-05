"""
STEP 1 — Build SigLIP RAG Knowledge Base
=========================================
Input sources:
  memecap  — memecap-data/memes-trainval.json + memes-test.json  (6382 entries)
  hmd      — captioned-hmd-data/captions_output{1,2,3}.jsonl     (1197 entries)

Output stores:
  --store memecap  →  pipeline/rag_store_memecap/
  --store full     →  pipeline/rag_store_full/      (memecap + hmd combined)
  --store both     →  builds both stores

Each store contains:
  embeddings.npy   float32 array (N, 1536)
  metadata.pkl     list of dicts, one per entry

Embedding strategy (multimodal):
  • SigLIP encodes each entry as BOTH an image embed and a text embed
    (each 768-d for siglip-base-patch16-224).
  • The two are L2-normalised and concatenated → 1536-d vector.
  • All stored vectors are L2-normalised so retrieval is a plain dot product
    (= cosine similarity).

Usage:
  python pipeline/build_siglip_rag.py                 # default: memecap only
  python pipeline/build_siglip_rag.py --store full     # memecap + HMD
  python pipeline/build_siglip_rag.py --store both     # build both stores
"""

import argparse
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE             = Path(__file__).resolve().parents[1]
MEMECAP_TRAINVAL = BASE / "memecap-data" / "memes-trainval.json"
MEMECAP_TEST     = BASE / "memecap-data" / "memes-test.json"
MEMECAP_IMG_DIR  = BASE / "memecap-data" / "memes"
HMD_JSONL_FILES  = [
    BASE / "captioned-hmd-data" / "captions_output1.jsonl",
    BASE / "captioned-hmd-data" / "captions_output2.jsonl",
    BASE / "captioned-hmd-data" / "captions_output3.jsonl",
]
HMD_IMG_DIR      = BASE / "facebook-data"   # img_fname is e.g. "img/42953.png"

STORE_DIRS = {
    "memecap": BASE / "pipeline" / "rag_store_memecap",
    "full":    BASE / "pipeline" / "rag_store_full",
}

SIGLIP_MODEL = "google/siglip-base-patch16-224"
BATCH_SIZE   = 32


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_memecap() -> list:
    items = []
    for path in [MEMECAP_TRAINVAL, MEMECAP_TEST]:
        with open(path) as f:
            data = json.load(f)
        batch = data if isinstance(data, list) else list(data.values())
        for item in batch:
            item.setdefault("_img_base", MEMECAP_IMG_DIR)
        items.extend(batch)
    return items


def load_hmd() -> list:
    items = []
    for path in HMD_JSONL_FILES:
        if not path.exists():
            print(f"  [warn] HMD file not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # img_fname in HMD is "img/42953.png"; resolve against HMD_IMG_DIR
                item.setdefault("_img_base", HMD_IMG_DIR)
                items.append(item)
    return items


# ── Text builder ──────────────────────────────────────────────────────────────

def build_text(item: dict) -> str:
    parts = []
    if item.get("title"):
        parts.append(item["title"])
    if item.get("meme_captions"):
        parts.extend(item["meme_captions"])
    if item.get("img_captions"):
        parts.extend(item["img_captions"])
    if item.get("metaphors"):
        for m in item["metaphors"]:
            if isinstance(m, dict) and m.get("metaphor") and m.get("meaning"):
                parts.append(f"{m['metaphor']}: {m['meaning']}")
    return " | ".join(parts)[:512]


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


# ── Embedding loop ────────────────────────────────────────────────────────────

def embed_entries(items: list, model, processor, device: str) -> tuple[np.ndarray, list]:
    embeddings = []
    metadata   = []

    for item in tqdm(items, desc="Encoding entries"):
        img_fname = item.get("img_fname", "")
        img_base  = item.get("_img_base", MEMECAP_IMG_DIR)
        img_path  = Path(img_base) / img_fname

        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            text  = build_text(item)

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs    = model(**inputs)
                img_embed  = outputs.image_embeds[0].cpu().float().numpy()
                text_embed = outputs.text_embeds[0].cpu().float().numpy()

            img_embed  = l2_norm(img_embed)
            text_embed = l2_norm(text_embed)
            combined   = l2_norm(np.concatenate([img_embed, text_embed]))

            embeddings.append(combined)
            metadata.append({
                "post_id":       item.get("post_id", ""),
                "img_fname":     img_fname,
                "img_path":      str(img_path),
                "title":         item.get("title", ""),
                "meme_captions": item.get("meme_captions", []),
                "img_captions":  item.get("img_captions", []),
                "metaphors":     item.get("metaphors", []),
                "text":          text,
            })

        except Exception as e:
            print(f"  [skip] {img_fname}: {e}")

    return np.array(embeddings, dtype=np.float32), metadata


# ── Build one store ───────────────────────────────────────────────────────────

def build_store(store_name: str, items: list, model, processor, device: str):
    store_dir = STORE_DIRS[store_name]
    store_dir.mkdir(parents=True, exist_ok=True)
    emb_path  = store_dir / "embeddings.npy"
    meta_path = store_dir / "metadata.pkl"

    print(f"\n[{store_name}] Embedding {len(items)} entries → {store_dir}")
    embeddings, metadata = embed_entries(items, model, processor, device)
    print(f"[{store_name}] Embedded {len(embeddings)} entries  (shape: {embeddings.shape})")

    np.save(emb_path, embeddings)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[{store_name}] Saved embeddings → {emb_path}")
    print(f"[{store_name}] Saved metadata   → {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build SigLIP RAG store(s)")
    parser.add_argument(
        "--store", default="memecap",
        choices=["memecap", "full", "both"],
        help=(
            "'memecap' = MemeCap only (default), "
            "'full' = MemeCap + captioned HMD, "
            "'both' = build both stores"
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1 — Building SigLIP RAG store(s)")
    print(f"  store = {args.store}")
    print("=" * 60)

    memecap_items = load_memecap()
    print(f"Loaded {len(memecap_items)} MemeCap entries")

    hmd_items = []
    if args.store in ("full", "both"):
        hmd_items = load_hmd()
        print(f"Loaded {len(hmd_items)} HMD entries")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SigLIP model: {SIGLIP_MODEL}  (device={device})")
    model     = SiglipModel.from_pretrained(SIGLIP_MODEL).to(device).eval()
    processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL)

    stores_to_build = []
    if args.store == "memecap":
        stores_to_build = [("memecap", memecap_items)]
    elif args.store == "full":
        stores_to_build = [("full", memecap_items + hmd_items)]
    else:  # both
        stores_to_build = [
            ("memecap", memecap_items),
            ("full",    memecap_items + hmd_items),
        ]

    for store_name, items in stores_to_build:
        build_store(store_name, items, model, processor, device)

    print("\nStep 1 complete.  Run pipeline/run_baseline.py next.")
    print("  Use --rag-store memecap or --rag-store full in run_baseline.py")


if __name__ == "__main__":
    main()
