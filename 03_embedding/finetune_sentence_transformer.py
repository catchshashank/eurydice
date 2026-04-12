"""
03_embedding/finetune_sentence_transformer.py

Fine-tune multi-qa-MiniLM-L6-cos-v1 on synthetic query–product pairs using
MultipleNegativesRankingLoss (paper §4.1.3). Saves the model and builds a
FAISS IVF-Flat index over the catalog (paper §4.1.4).

Usage:
    python 03_embedding/finetune_sentence_transformer.py
    python 03_embedding/finetune_sentence_transformer.py --epochs 3 --batch-size 64
"""

import argparse
import json
import logging
import math
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(jsonl_path: str) -> tuple[list[str], list[str]]:
    queries, products = [], []
    seen_queries, seen_products = set(), set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["query"]
            p = row["product_text"]
            if q not in seen_queries and p not in seen_products:
                queries.append(q)
                products.append(p)
                seen_queries.add(q)
                seen_products.add(p)
    return queries, products


def finetune(queries, products, base_model, output_dir, epochs, batch_size):
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[q, p]) for q, p in zip(queries, products)]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = MultipleNegativesRankingLoss(model)

    warmup = math.ceil(len(loader) * epochs * 0.1)
    log.info(f"Training: {len(examples)} pairs, batch={batch_size}, epochs={epochs}, warmup={warmup}")

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
        output_path=output_dir,
    )
    return model


def build_faiss_index(model, catalog_path: str, index_path: str, id_map_path: str):
    import faiss
    import numpy as np

    log.info("Loading catalog and computing product embeddings ...")
    catalog = []
    with open(catalog_path, encoding="utf-8") as f:
        for line in f:
            catalog.append(json.loads(line))

    texts = [p["product_text"] for p in catalog]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    n = len(embeddings)
    n_lists = max(1, int(math.sqrt(n)))

    log.info(f"Building IVF-Flat index: {n} vectors, dim={dim}, n_lists={n_lists}")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add_with_ids(embeddings, np.arange(n, dtype="int64"))

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    log.info(f"Saved FAISS index to {index_path}")

    id_map = {i: p["parent_asin"] for i, p in enumerate(catalog)}
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)
    log.info(f"Saved ID map to {id_map_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--data", default=None, help="Path to synthetic_queries.jsonl")
    parser.add_argument("--output", default=None, help="Path to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true", help="Skip FAISS index build")
    args = parser.parse_args()

    cfg = load_config(args.config)
    emb_cfg = cfg["embedding"]

    data_path  = args.data       or cfg["paths"]["synthetic_queries"]
    output_dir = args.output     or cfg["paths"]["embedding_model"]
    epochs     = args.epochs     or emb_cfg["epochs"]
    batch_size = args.batch_size or emb_cfg["batch_size"]
    base_model = emb_cfg["base_model"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"Loading pairs from {data_path} ...")
    queries, products = load_pairs(data_path)
    log.info(f"Loaded {len(queries)} deduplicated pairs.")

    model = finetune(queries, products, base_model, output_dir, epochs, batch_size)
    log.info(f"Model saved to {output_dir}")

    if not args.skip_index:
        build_faiss_index(
            model,
            catalog_path=cfg["paths"]["catalog"],
            index_path=cfg["paths"]["faiss_index"],
            id_map_path=cfg["paths"]["faiss_id_map"],
        )


if __name__ == "__main__":
    main()
