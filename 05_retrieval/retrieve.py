"""
05_retrieval/retrieve.py

Given a natural-language query:
  1. Extract structured filters with the fine-tuned Flan-T5-small
  2. Preselect product IDs whose metadata satisfies those filters
  3. Run FAISS inner-product search restricted to that subset
  4. Return ranked top-k products

Usage (interactive demo):
    python 05_retrieval/retrieve.py
    python 05_retrieval/retrieve.py --query "blue linen trousers under $40 with strong reviews"
    python 05_retrieval/retrieve.py --top-k 5
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PREFIX = "extract filters: "


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_catalog(path: str) -> list[dict]:
    catalog = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            catalog.append(json.loads(line))
    return catalog


def resolve_qualitative(value: str, thresholds: dict, metric: str) -> tuple[float | None, float | None]:
    if value not in thresholds.get(metric, {}):
        return None, None
    lo, hi = thresholds[metric][value]
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


def apply_filters(catalog: list[dict], filters: dict, thresholds: dict) -> list[int]:
    matching_ids = []
    for idx, product in enumerate(catalog):
        if _matches(product, filters, thresholds):
            matching_ids.append(idx)
    return matching_ids


def _matches(product: dict, filters: dict, thresholds: dict) -> bool:
    price = product.get("price")
    rating = product.get("average_rating")
    review_count = product.get("rating_number")
    subcategory = product.get("subcategory", "")

    def check_numeric(value, min_val, max_val):
        if value is None:
            return True  # missing metadata: don't filter out
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    # Price
    p_min = filters.get("price_min")
    p_max = filters.get("price_max")
    if not check_numeric(price, p_min, p_max):
        return False

    # Rating
    r_min = filters.get("average_rating_min")
    r_max = filters.get("average_rating_max")
    if not check_numeric(rating, r_min, r_max):
        return False

    # Review count — may be qualitative
    rc_min = filters.get("review_count_min")
    rc_max = filters.get("review_count_max")
    if isinstance(rc_min, str):
        rc_min, _ = resolve_qualitative(rc_min, thresholds, "review_count")
    if isinstance(rc_max, str):
        _, rc_max = resolve_qualitative(rc_max, thresholds, "review_count")
    if not check_numeric(review_count, rc_min, rc_max):
        return False

    # Subcategory
    filt_sub = filters.get("subcategory")
    if filt_sub and filt_sub.lower() not in subcategory.lower():
        return False

    return True


def extract_filters(query: str, tokenizer, model) -> dict:
    import torch
    inputs = tokenizer(INPUT_PREFIX + query, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        log.warning(f"Filter model returned non-dict JSON: {text!r}")
        return {}
    except json.JSONDecodeError:
        log.warning(f"Could not parse filter JSON: {text!r}")
        return {}


def retrieve(query: str, catalog, embedding_model, faiss_index, id_map, tokenizer, filter_model, thresholds, top_k):
    filters = extract_filters(query, tokenizer, filter_model)
    log.info(f"Extracted filters: {filters}")

    matching_ids = apply_filters(catalog, filters, thresholds)
    log.info(f"{len(matching_ids)} products pass filters (out of {len(catalog)})")

    if not matching_ids:
        log.warning("No products match the filters — returning unfiltered results.")
        matching_ids = list(range(len(catalog)))

    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search all products then post-filter to matching IDs (efficient for small catalogs)
    faiss_index.nprobe = 10
    n_search = len(catalog)
    scores, indices = faiss_index.search(query_embedding, n_search)

    matching_set = set(matching_ids)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if int(idx) not in matching_set:
            continue
        asin = id_map.get(str(idx), id_map.get(idx))
        results.append({"rank": len(results) + 1, "parent_asin": asin, "score": float(score)})
        if len(results) >= top_k:
            break
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    cfg = load_config(args.config)
    top_k = args.top_k or cfg["retrieval"]["top_k"]
    thresholds = cfg["thresholds"]

    log.info("Loading models and index ...")
    embedding_model = SentenceTransformer(cfg["paths"]["embedding_model"])
    faiss_index = faiss.read_index(cfg["paths"]["faiss_index"])
    with open(cfg["paths"]["faiss_id_map"]) as f:
        id_map = json.load(f)
    catalog = load_catalog(cfg["paths"]["catalog"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["filter_model"])
    filter_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["paths"]["filter_model"])

    query = args.query
    if not query:
        query = input("Enter search query: ").strip()

    results = retrieve(query, catalog, embedding_model, faiss_index, id_map, tokenizer, filter_model, thresholds, top_k)

    print(f"\nTop-{top_k} results for: \"{query}\"")
    print("-" * 60)
    for r in results:
        asin = r["parent_asin"]
        product = next((p for p in catalog if p["parent_asin"] == asin), {})
        print(f"  {r['rank']:2d}. [{r['score']:.4f}] {product.get('title', asin)[:80]}")


if __name__ == "__main__":
    main()
