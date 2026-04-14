"""
06_evaluation/evaluate.py

Compute precision@k and recall@k for the retrieval system against manually
annotated test queries (paper §5). Writes eval_results.json.

test_queries.jsonl schema (one line per query):
    {
      "query": "...",
      "relevant_asins": ["B0...", ...]   # ground-truth positive product IDs
    }

Usage:
    python 06_evaluation/evaluate.py
    python 06_evaluation/evaluate.py --top-k 10
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--test-queries", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Import retrieval logic
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "05_retrieval"))
    from retrieve import retrieve, load_catalog  # noqa: E402

    cfg = load_config(args.config)
    k_values = cfg["evaluation"]["k_values"]
    top_k = args.top_k or max(k_values)
    thresholds = cfg["thresholds"]

    test_path = args.test_queries or cfg["paths"]["test_queries"]
    out_path  = Path(cfg["paths"]["eval_results"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading models and index ...")
    embedding_model = SentenceTransformer(cfg["paths"]["embedding_model"])
    faiss_index = faiss.read_index(cfg["paths"]["faiss_index"])
    with open(cfg["paths"]["faiss_id_map"]) as f:
        id_map = json.load(f)
    catalog = load_catalog(cfg["paths"]["catalog"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["filter_model"])
    filter_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["paths"]["filter_model"])

    test_queries = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            test_queries.append(json.loads(line))
    log.info(f"Evaluating {len(test_queries)} test queries ...")

    # Accumulate P@k and R@k
    agg = {k: {"precision": 0.0, "recall": 0.0} for k in k_values}

    for i, item in enumerate(test_queries):
        query = item["query"]
        relevant = set(item["relevant_asins"])
        results = retrieve(query, catalog, embedding_model, faiss_index, id_map,
                           tokenizer, filter_model, thresholds, top_k)
        retrieved_asins = [r["parent_asin"] for r in results]

        for k in k_values:
            agg[k]["precision"] += precision_at_k(retrieved_asins, relevant, k)
            agg[k]["recall"]    += recall_at_k(retrieved_asins, relevant, k)

        if (i + 1) % 10 == 0:
            log.info(f"Processed {i + 1}/{len(test_queries)} queries ...")

    n = len(test_queries)
    results_out = {}
    print(f"\n{'k':>4}  {'Precision@k':>12}  {'Recall@k':>10}")
    print("-" * 32)
    for k in k_values:
        p = agg[k]["precision"] / n
        r = agg[k]["recall"] / n
        results_out[k] = {"precision": round(p, 4), "recall": round(r, 4)}
        print(f"{k:>4}  {p:>12.4f}  {r:>10.4f}")

    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
