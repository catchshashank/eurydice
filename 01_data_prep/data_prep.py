"""
01_data_prep/data_prep.py

Sample 100 clothing products from meta_Amazon_Fashion.jsonl.gz, clean text fields,
and assign subcategory labels via Gemini Flash. Writes data/processed/catalog.jsonl.

Usage:
    python 01_data_prep/data_prep.py
    python 01_data_prep/data_prep.py --config configs/data_config.yaml --seed 42
"""

import argparse
import gzip
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

from google import genai
import yaml
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"https?://\S+", " ", text)      # strip URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # strip non-ASCII
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_product_text(record: dict) -> str:
    parts = [
        record.get("title", ""),
        " ".join(record.get("features", []) or []),
        " ".join(record.get("description", []) or []),
        json.dumps(record.get("details", {}) or {}),
    ]
    return clean_text(" ".join(parts))


def sample_products(gz_path: str, n: int, seed: int) -> list[dict]:
    log.info(f"Streaming {gz_path} to reservoir-sample {n} products ...")
    reservoir = []
    rng = random.Random(seed)
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if i < n:
                reservoir.append(record)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = record
    log.info(f"Sampled {len(reservoir)} products from {i + 1} total records.")
    return reservoir


def assign_subcategories(products: list[dict], subcategories: list[str], api_key: str) -> list[dict]:
    client = genai.Client(api_key=api_key)
    cats_str = ", ".join(subcategories)

    lines = []
    for p in products:
        title = p.get("title", "")
        features = " ".join(p.get("features", []) or [])[:200]
        lines.append(f'{p["parent_asin"]}: {title} | {features}')
    products_block = "\n".join(lines)

    prompt = (
        f"Classify each clothing product into exactly one of these subcategories: {cats_str}.\n"
        f"Reply with one line per product in the format: parent_asin<TAB>subcategory\n"
        f"Use only the exact subcategory names listed above. No extra text.\n\n"
        f"{products_block}"
    )

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            break
        except Exception as e:
            wait = 60 * (attempt + 1)
            log.warning(f"Gemini error (attempt {attempt + 1}/{MAX_RETRIES}): {e}\nRetrying in {wait}s ...")
            time.sleep(wait)
    else:
        log.error("All Gemini retries failed. Defaulting all subcategories.")
        for p in products:
            p["subcategory"] = subcategories[0]
        return products

    # Parse response
    asin_to_sub = {}
    for line in response.text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            asin, sub = parts[0].strip(), parts[1].strip()
            if sub in subcategories:
                asin_to_sub[asin] = sub

    for p in products:
        p["subcategory"] = asin_to_sub.get(p["parent_asin"], subcategories[0])
        log.info(f"  {p['parent_asin']}: {p['subcategory']}")

    return products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-subcategory", action="store_true",
                        help="Skip Gemini subcategory assignment (for testing)")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    raw_path = cfg["paths"]["raw_metadata"]
    out_path = Path(cfg["paths"]["catalog"])
    n = cfg["catalog"]["n_products"]
    subcategories = cfg["subcategories"]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    products = sample_products(raw_path, n, args.seed)

    for p in products:
        p["product_text"] = build_product_text(p)

    if not args.skip_subcategory:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            log.error("GEMINI_API_KEY not set. Run with --skip-subcategory or set the key in .env")
            sys.exit(1)
        log.info("Assigning subcategory labels via Gemini Flash ...")
        products = assign_subcategories(products, subcategories, api_key)
    else:
        log.warning("Skipping subcategory assignment.")

    with open(out_path, "w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    log.info(f"Wrote {len(products)} products to {out_path}")


if __name__ == "__main__":
    main()
