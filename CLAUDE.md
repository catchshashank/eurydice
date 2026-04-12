# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Replication pilot of Siddiqui et al. (2026) "LLM-based Semantic Search for Conversational Queries in E-commerce" (`siddiqui et al (2026) - llm-based semantic search e-commerce.pdf`), adapted for **100 Amazon clothing products** instead of the paper's 1.3M cell phone products.

## Environment Setup

```bash
pip install -r requirements.txt
# Copy .env.example to .env and fill in API keys (never commit .env)
```

- Python 3.11
- API keys needed: Gemini Flash (synthetic query generation)

## Pipeline Run Order

Scripts run sequentially; each stage produces outputs consumed by the next.

```
01_data_prep → 02_synthetic_data → 03_embedding → 04_filter_extractor → 05_retrieval → 06_evaluation
```

| Script | Purpose | Paper §|
|--------|---------|--------|
| `01_data_prep` | Sample 100 clothing products from `meta_Amazon_Fashion.jsonl.gz`; clean text (strip HTML, lowercase, trim); reassign subcategory labels via Gemini Flash | §3 |
| `02_synthetic_data` | Generate ~10 synthetic queries per product via Gemini Flash zero-shot prompt; enrich queries with price/rating/review constraints; extract structured filters as JSON | §4.1.2, §4.2.1 |
| `03_embedding` | Fine-tune `multi-qa-MiniLM-L6-cos-v1` on query–product pairs using `MultipleNegativesRankingLoss` (batch size 168); build FAISS IVF-Flat index over product embeddings | §4.1.3, §4.1.4 |
| `04_filter_extractor` | Fine-tune `Flan-T5-small` with `Seq2SeqTrainer` on enriched-query→structured-filter pairs; outputs JSON with `price_min/max`, `review_count_min/max`, `average_rating_min/max`, `subcategory` | §4.2.2 |
| `05_retrieval` | At query time: extract structured filters (Flan-T5-small), preselect matching product IDs, run FAISS inner-product similarity search over filtered subset, return ranked top-k | §4.1.4 |
| `06_evaluation` | Compute precision@k and recall@k (k=1,2,3,5,10) against manually annotated query–product pairs | §5 |

## Architecture

The framework has two components that combine at retrieval time:

**Embedding component** — fine-tuned `multi-qa-MiniLM-L6-cos-v1` encodes both products (indexed in FAISS) and user queries into a shared dense vector space. Fine-tuning uses LLM-generated synthetic query–product pairs so the space aligns product descriptions with how users would search for them.

**Structure component** — fine-tuned `Flan-T5-small` converts a natural-language query into a structured JSON filter. Filters are applied *before* FAISS search to preselect candidate product IDs, then FAISS ranks the filtered subset by cosine similarity.

The join key between reviews (`Amazon_Fashion.jsonl.gz`) and metadata (`meta_Amazon_Fashion.jsonl.gz`) is `parent_asin`.

## Key Deviations from Paper

| Aspect | Paper | This pilot |
|--------|-------|------------|
| Domain | Cell Phones & Accessories | Clothing |
| Scale | 1.3M products | 100 products |
| Subcategory schema | Cell Phones / Cell Phone Accessories | Adapted for clothing taxonomy |
| Qualitative thresholds (Table 2) | Fixed values in paper | See `configs/data_config.yaml` |

## Structured Filter Schema

Filters output by Flan-T5-small (adapted from paper Table 1):

```json
{
  "price_min": null,
  "price_max": 300.0,
  "review_count_min": "high",
  "review_count_max": null,
  "average_rating_min": null,
  "average_rating_max": null,
  "subcategory": "<clothing subcategory>"
}
```

Qualitative values (`"low"`, `"medium"`, `"high"`) are mapped to numeric thresholds at retrieval time via `configs/data_config.yaml`. The paper's original thresholds (for reference):

| Metric | Low | Medium | High |
|--------|-----|--------|------|
| Rating | [0, 4.0) | [4.0, 5] | [4.5, 5] |
| #Reviews | [0, 100) | [100, ∞) | [1000, ∞) |

## Synthetic Query Generation Prompt

The Gemini Flash prompt (paper Figure 2) instructs the model to generate natural-language queries per product based on its `parent_asin`, title, features, description, and technical specifications. Multiple products can be batched in a single prompt. Each product receives ~10 diverse queries covering technical specs, user needs, and general intent.

## Evaluation

- Metrics: precision@k and recall@k for k ∈ {1, 2, 3, 5, 10}
- Test set: manually annotated query–product relevance pairs (adapted from paper's ESCI-based approach)
- Paper's best results (ours system): precision@1=0.32, recall@10=0.57
