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
    """Load query-product pairs, deduplicating on query only (products repeat across queries)."""
    queries, products = [], []
    seen_queries = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["query"]
            p = row["product_text"]
            if q not in seen_queries:
                queries.append(q)
                products.append(p)
                seen_queries.add(q)
    return queries, products


def finetune(queries, products, base_model, output_dir, epochs, batch_size):
    import os
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["USE_TORCH"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import random
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer, models as st_models
    from torch.optim import AdamW

    HF_NAME = f"sentence-transformers/{base_model}" if "/" not in base_model else base_model
    log.info(f"Loading tokenizer and model from {HF_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    transformer = AutoModel.from_pretrained(HF_NAME)
    device = torch.device("cpu")
    transformer.to(device)

    optimizer = AdamW(transformer.parameters(), lr=2e-5)
    effective_batch = min(batch_size, len(queries))
    log.info(f"Training: {len(queries)} pairs, batch={effective_batch}, epochs={epochs}")

    def encode_texts(texts):
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = transformer(**enc)
        token_emb = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=1)

    pairs = list(zip(queries, products))
    transformer.train()
    for epoch in range(epochs):
        random.shuffle(pairs)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(pairs), effective_batch):
            batch = pairs[i:i + effective_batch]
            if len(batch) < 2:
                continue
            q_batch, p_batch = zip(*batch)
            q_emb = encode_texts(list(q_batch))
            p_emb = encode_texts(list(p_batch))
            scores = q_emb @ p_emb.T
            labels = torch.arange(len(q_batch), device=device)
            loss_val = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            epoch_loss += loss_val.item()
            n_batches += 1
            log.info(f"  epoch {epoch+1} batch {n_batches} loss={loss_val.item():.4f}")
        log.info(f"Epoch {epoch+1}/{epochs} avg_loss={epoch_loss/max(1,n_batches):.4f}")

    # Build SentenceTransformer by copying trained weights (avoids Windows safetensors file-lock)
    word_emb = st_models.Transformer(HF_NAME, max_seq_length=128)
    word_emb.auto_model.load_state_dict(transformer.state_dict())
    pooling = st_models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[word_emb, pooling], device="cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    st_model.save(output_dir)
    log.info(f"SentenceTransformer saved to {output_dir}")
    return st_model


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
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true")
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
    log.info(f"Loaded {len(queries)} pairs.")

    model = finetune(queries, products, base_model, output_dir, epochs, batch_size)

    if not args.skip_index:
        build_faiss_index(
            model,
            catalog_path=cfg["paths"]["catalog"],
            index_path=cfg["paths"]["faiss_index"],
            id_map_path=cfg["paths"]["faiss_id_map"],
        )


if __name__ == "__main__":
    main()
