import json
import logging
import numpy as np
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PREFIX = "extract filters: "
state = {}


def load_catalog(path):
    catalog = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            catalog.append(json.loads(line))
    return catalog


def resolve_qualitative(value, thresholds, metric):
    lo, hi = thresholds.get(metric, {}).get(value, (None, None))
    return (float(lo) if lo is not None else None,
            float(hi) if hi is not None else None)


def _matches(product, filters, thresholds):
    def check(val, lo, hi):
        if val is None:
            return True
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    if not check(product.get("price"), filters.get("price_min"), filters.get("price_max")):
        return False
    if not check(product.get("average_rating"), filters.get("average_rating_min"), filters.get("average_rating_max")):
        return False

    rc_min = filters.get("review_count_min")
    rc_max = filters.get("review_count_max")
    if isinstance(rc_min, str):
        rc_min, _ = resolve_qualitative(rc_min, thresholds, "review_count")
    if isinstance(rc_max, str):
        _, rc_max = resolve_qualitative(rc_max, thresholds, "review_count")
    if not check(product.get("rating_number"), rc_min, rc_max):
        return False

    filt_sub = filters.get("subcategory")
    if filt_sub and filt_sub.lower() not in (product.get("subcategory") or "").lower():
        return False

    return True


def apply_filters(catalog, filters, thresholds):
    return [i for i, p in enumerate(catalog) if _matches(p, filters, thresholds)]


def extract_filters(query):
    import torch
    tokenizer, model = state["tokenizer"], state["filter_model"]
    inputs = tokenizer(INPUT_PREFIX + query, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if text.startswith('"') or text.startswith("'"):
        text = "{" + text
    if text and not text.endswith("}"):
        text = text + "}"
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


@asynccontextmanager
async def lifespan(app):
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    log.info("Downloading models from HF Hub...")
    model_dir = snapshot_download("catchshashank/eurydice-models")

    log.info("Loading embedding model...")
    state["embedding_model"] = SentenceTransformer(f"{model_dir}/embedding")

    log.info("Loading FAISS index...")
    state["faiss_index"] = faiss.read_index(f"{model_dir}/faiss.index")
    with open(f"{model_dir}/faiss_id_map.json") as f:
        state["id_map"] = json.load(f)

    log.info("Loading filter extractor...")
    state["tokenizer"] = AutoTokenizer.from_pretrained(f"{model_dir}/filter_extractor")
    state["filter_model"] = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}/filter_extractor")

    with open("configs/data_config.yaml") as f:
        cfg = yaml.safe_load(f)
    state["thresholds"] = cfg["thresholds"]
    state["catalog"] = load_catalog("data/processed/catalog.jsonl")
    state["top_k"] = 5

    log.info("Ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://catchshashank.github.io"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(state)}


@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    if not state:
        raise HTTPException(status_code=503, detail="Models still loading")

    catalog = state["catalog"]
    thresholds = state["thresholds"]
    top_k = state["top_k"]
    id_map = state["id_map"]

    filters = extract_filters(q)
    log.info(f"Query: {q!r} → filters: {filters}")

    matching_ids = apply_filters(catalog, filters, thresholds) or list(range(len(catalog)))

    query_emb = np.array(
        state["embedding_model"].encode([q], normalize_embeddings=True), dtype="float32"
    )
    state["faiss_index"].nprobe = 10
    scores, indices = state["faiss_index"].search(query_emb, len(catalog))

    matching_set = set(matching_ids)
    results = []
    for score, raw_idx in zip(scores[0], indices[0]):
        if raw_idx == -1 or int(raw_idx) not in matching_set:
            continue
        asin = id_map.get(str(raw_idx)) or id_map.get(raw_idx)
        product = next((p for p in catalog if p["parent_asin"] == asin), None)
        if not product:
            continue
        imgs = product.get("images", [])
        img = next((i["large"] for i in imgs if i.get("variant") == "MAIN" and i.get("large")), None)
        if not img and imgs:
            img = imgs[0].get("large")
        results.append({
            "asin": asin,
            "title": product.get("title"),
            "rating": product.get("average_rating"),
            "reviews": product.get("rating_number"),
            "price": product.get("price"),
            "subcategory": product.get("subcategory"),
            "image": img,
            "score": float(score),
        })
        if len(results) >= top_k:
            break

    return {"query": q, "filters": filters, "results": results}
