"""
04_filter_extractor/finetune_flan_t5.py

Fine-tune Flan-T5-small as a seq2seq filter extractor (paper §4.2.2).
Input:  enriched natural-language query
Output: structured JSON filter string

Usage:
    python 04_filter_extractor/finetune_flan_t5.py
    python 04_filter_extractor/finetune_flan_t5.py --epochs 5
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PREFIX = "extract filters: "


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_filter_pairs(jsonl_path: str) -> list[dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_hf_dataset(rows: list[dict], tokenizer, max_input=256, max_target=256):
    from datasets import Dataset

    inputs, targets = [], []
    for row in rows:
        q = INPUT_PREFIX + row["enriched_query"]
        label = json.dumps(row["structured_filters"], ensure_ascii=False)
        inputs.append(q)
        targets.append(label)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input"], max_length=max_input, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = Dataset.from_dict({"input": inputs, "target": targets})
    return ds.map(tokenize, batched=True, remove_columns=["input", "target"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

    cfg = load_config(args.config)
    fe_cfg = cfg["filter_extractor"]

    data_path  = args.data       or cfg["paths"]["filter_pairs"]
    output_dir = args.output     or cfg["paths"]["filter_model"]
    epochs     = args.epochs     or fe_cfg["epochs"]
    batch_size = args.batch_size or fe_cfg["batch_size"]
    base_model = fe_cfg["base_model"]
    eval_split = fe_cfg["eval_split"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = load_filter_pairs(data_path)
    log.info(f"Loaded {len(rows)} filter pairs from {data_path}")

    split_idx = max(1, int(len(rows) * (1 - eval_split)))
    train_rows, eval_rows = rows[:split_idx], rows[split_idx:]
    log.info(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    train_ds = build_hf_dataset(train_rows, tokenizer)
    eval_ds  = build_hf_dataset(eval_rows,  tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        dataloader_num_workers=0,   # avoid multiprocessing crash on Windows
        use_cpu=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting Flan-T5-small fine-tuning ...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Filter extractor saved to {output_dir}")


if __name__ == "__main__":
    main()
