"""
STEP 1 — Build SigLIP RAG Knowledge Base from MemeCap data
===========================================================
Input:  memecap-data/memes-trainval.json  (5823 items)
        memecap-data/memes-test.json       (559  items)
        memecap-data/memes/               (images)

Output: pipeline/rag_store/embeddings.npy   float32 array (N, 1536)
        pipeline/rag_store/metadata.pkl     list of dicts, one per entry

Embedding strategy (multimodal):
  • SigLIP encodes each MemeCap entry as BOTH an image embed and a text embed
    (each 768-d for siglip-base-patch16-224).
  • The two are L2-normalised and concatenated → 1536-d vector.
  • All stored vectors are L2-normalised so retrieval is a plain dot product
    (= cosine similarity).

At query time (step 2) the same encoding is applied to a Facebook meme
(its image + overlay text) and the top-k nearest MemeCap entries are
returned as RAG context for the Qwen3-VL model.
"""

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
RAG_STORE        = BASE / "pipeline" / "rag_store"
RAG_STORE.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PATH  = RAG_STORE / "embeddings.npy"
METADATA_PATH    = RAG_STORE / "metadata.pkl"

SIGLIP_MODEL     = "google/siglip-base-patch16-224"
BATCH_SIZE       = 32   # reduce if GPU OOM


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_memecap(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


def build_text(item: dict) -> str:
    """
    Flatten a MemeCap entry into a single string that captures its meaning.
    Field priority: meme_captions > title > img_captions > metaphors.
    """
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
    return " | ".join(parts)[:512]   # SigLIP max token budget


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


# ── Embedding loop ────────────────────────────────────────────────────────────

def embed_entries(items: list, model, processor, device: str) -> tuple[np.ndarray, list]:
    embeddings = []
    metadata   = []

    for item in tqdm(items, desc="Encoding MemeCap entries"):
        img_fname = item.get("img_fname", "")
        img_path  = MEMECAP_IMG_DIR / img_fname

        if not img_path.exists():
            # skip entries whose image is missing
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
                outputs     = model(**inputs)
                img_embed   = outputs.image_embeds[0].cpu().float().numpy()
                text_embed  = outputs.text_embeds[0].cpu().float().numpy()

            # L2-normalise each modality before concatenation
            img_embed  = l2_norm(img_embed)
            text_embed = l2_norm(text_embed)
            combined   = l2_norm(np.concatenate([img_embed, text_embed]))   # 1536-d

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
            continue

    return np.array(embeddings, dtype=np.float32), metadata


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1 — Building SigLIP RAG from MemeCap data")
    print("=" * 60)

    trainval = load_memecap(MEMECAP_TRAINVAL)
    test     = load_memecap(MEMECAP_TEST)
    all_items = trainval + test
    print(f"Loaded {len(trainval)} trainval + {len(test)} test = {len(all_items)} MemeCap entries")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SigLIP model: {SIGLIP_MODEL}  (device={device})")
    model     = SiglipModel.from_pretrained(SIGLIP_MODEL).to(device).eval()
    processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL)

    embeddings, metadata = embed_entries(all_items, model, processor, device)

    print(f"\nEmbedded {len(embeddings)} entries  (shape: {embeddings.shape})")

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved embeddings → {EMBEDDINGS_PATH}")
    print(f"Saved metadata   → {METADATA_PATH}")
    print("\nStep 1 complete.  Run step2_run_baseline.py next.")


if __name__ == "__main__":
    main()