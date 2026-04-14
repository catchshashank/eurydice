"""
02_synthetic_data/generate_queries.py

Stage 1 — Raw synthetic queries: generate ~10 natural-language queries per product
using Gemini Flash zero-shot prompting (paper §4.1.2). Writes synthetic_queries.jsonl.

Stage 2 — Enriched queries + structured filters: enrich each query with price/
rating/review constraints, then extract structured JSON filters (paper §4.2.1).
Writes enriched_queries.jsonl and filter_pairs.jsonl.

Usage:
    python 02_synthetic_data/generate_queries.py
    python 02_synthetic_data/generate_queries.py --stage raw
    python 02_synthetic_data/generate_queries.py --stage enrich
    python 02_synthetic_data/generate_queries.py --stage both   (default)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from google import genai
import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_QUERY_PROMPT = """\
For each product described below, generate natural language search queries that a user \
might input in a clothing e-commerce search engine to find that product.

The queries should capture various features (e.g., material, fit, style), benefits \
(e.g., comfort, durability), and user needs (e.g., occasion, size compatibility), \
using synonyms and different phrasings. Aim for a variety of specific and general queries.

Example queries: "I am looking for a product with [feature]" or \
"Find a product that offers [benefit]."

Each response should only indicate the product's parent_asin, followed by its queries. \
Do not include any other section, titles, categories, or headings. \
Output the queries in simple bullet points.

<PRODUCTS>
{products_block}
"""

ENRICH_PROMPT_BATCH = """\
You are enriching clothing product search queries with shopping constraints.

For EACH query below, add one or more constraints naturally (price, rating, review count).
Return a JSON array — one object per query — with exactly these fields:
{{
  "original_query": "<original>",
  "enriched_query": "<query with constraints added naturally>",
  "structured_filters": {{
    "price_min": <number or null>,
    "price_max": <number or null>,
    "review_count_min": <"low"|"medium"|"high"|null>,
    "review_count_max": <"low"|"medium"|"high"|null>,
    "average_rating_min": <number or null>,
    "average_rating_max": <number or null>,
    "subcategory": "<subcategory or null>"
  }}
}}

Output ONLY the JSON array. No markdown, no extra text.

Queries (subcategory in parentheses):
{queries_block}
"""


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_catalog(path: str) -> list[dict]:
    products = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            products.append(json.loads(line))
    return products


def build_products_block(batch: list[dict]) -> str:
    lines = []
    for p in batch:
        lines.append(f"parent_asin: {p['parent_asin']}")
        lines.append(f"title: {p.get('title', '')}")
        lines.append(f"features: {' '.join(p.get('features', []) or [])}")
        lines.append(f"description: {' '.join(p.get('description', []) or [])}")
        lines.append("")
    return "\n".join(lines)


def parse_raw_queries(text: str, valid_asins: set) -> list[dict]:
    pairs = []
    current_asin = None
    for line in text.splitlines():
        line = line.strip()
        # Match bare ASIN line (with or without "parent_asin:" prefix)
        candidate = line.removeprefix("parent_asin:").strip()
        if candidate in valid_asins:
            current_asin = candidate
        elif line.startswith(("- ", "* ", "• ")) and current_asin:
            query = line[2:].strip()
            if query:
                pairs.append({"parent_asin": current_asin, "query": query})
    return pairs


def generate_raw_queries(catalog: list[dict], client, batch_size: int, queries_per_product: int) -> list[dict]:
    all_pairs = []
    valid_asins = {p["parent_asin"] for p in catalog}
    asin_to_text = {p["parent_asin"]: p.get("product_text", "") for p in catalog}

    for i in range(0, len(catalog), batch_size):
        batch = catalog[i:i + batch_size]
        log.info(f"Generating queries for products {i + 1}–{i + len(batch)} ...")
        prompt = RAW_QUERY_PROMPT.format(products_block=build_products_block(batch))
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            pairs = parse_raw_queries(response.text, valid_asins)
            for pair in pairs:
                pair["product_text"] = asin_to_text[pair["parent_asin"]]
            all_pairs.extend(pairs)
        except Exception as e:
            log.warning(f"Gemini error on batch {i}: {e}")
            time.sleep(5)

    log.info(f"Generated {len(all_pairs)} raw query–product pairs.")
    return all_pairs


def enrich_queries(pairs: list[dict], catalog: list[dict], client, batch_size: int = 50) -> tuple[list[dict], list[dict]]:
    asin_to_subcategory = {p["parent_asin"]: p.get("subcategory", "") for p in catalog}
    asin_to_text = {p["parent_asin"]: p.get("product_text", "") for p in catalog}
    enriched_rows = []
    filter_rows = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        queries_block = "\n".join(
            f"{j+1}. {p['query']} ({asin_to_subcategory.get(p['parent_asin'], '')})"
            for j, p in enumerate(batch)
        )
        prompt = ENRICH_PROMPT_BATCH.format(queries_block=queries_block)
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0].strip()
            results = json.loads(text)
            for pair, item in zip(batch, results):
                enriched_rows.append({
                    "parent_asin": pair["parent_asin"],
                    "product_text": asin_to_text.get(pair["parent_asin"], pair.get("product_text", "")),
                    "original_query": pair["query"],
                    "enriched_query": item.get("enriched_query", pair["query"]),
                })
                filter_rows.append({
                    "parent_asin": pair["parent_asin"],
                    "enriched_query": item.get("enriched_query", pair["query"]),
                    "structured_filters": item.get("structured_filters", {}),
                })
        except Exception as e:
            log.warning(f"Enrich error on batch {i}–{i+len(batch)}: {e}")
            # Fall back: keep original queries without enrichment
            for pair in batch:
                enriched_rows.append({
                    "parent_asin": pair["parent_asin"],
                    "product_text": asin_to_text.get(pair["parent_asin"], pair.get("product_text", "")),
                    "original_query": pair["query"],
                    "enriched_query": pair["query"],
                })
                filter_rows.append({
                    "parent_asin": pair["parent_asin"],
                    "enriched_query": pair["query"],
                    "structured_filters": {},
                })

        log.info(f"Enriched {min(i + batch_size, len(pairs))}/{len(pairs)} queries ...")

    return enriched_rows, filter_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--stage", choices=["raw", "enrich", "both"], default="both")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set in .env")
        sys.exit(1)

    cfg = load_config(args.config)
    client = genai.Client(api_key=api_key)

    catalog_path = cfg["paths"]["catalog"]
    catalog = load_catalog(catalog_path)
    log.info(f"Loaded {len(catalog)} products from {catalog_path}")

    syn_path = Path(cfg["paths"]["synthetic_queries"])
    enr_path = Path(cfg["paths"]["enriched_queries"])
    flt_path = Path(cfg["paths"]["filter_pairs"])
    syn_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stage in ("raw", "both"):
        pairs = generate_raw_queries(
            catalog,
            client,
            batch_size=cfg["synthetic"]["batch_size"],
            queries_per_product=cfg["synthetic"]["queries_per_product"],
        )
        with open(syn_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(pairs)} pairs to {syn_path}")
    else:
        pairs = []
        with open(syn_path, encoding="utf-8") as f:
            for line in f:
                pairs.append(json.loads(line))
        log.info(f"Loaded {len(pairs)} existing raw pairs from {syn_path}")

    if args.stage in ("enrich", "both"):
        enriched, filter_pairs = enrich_queries(pairs, catalog, client)
        with open(enr_path, "w", encoding="utf-8") as f:
            for r in enriched:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(flt_path, "w", encoding="utf-8") as f:
            for r in filter_pairs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(enriched)} enriched pairs to {enr_path}")
        log.info(f"Wrote {len(filter_pairs)} filter pairs to {flt_path}")


if __name__ == "__main__":
    main()
