# Companion data and model training tools

This directory contains selected code from Vinay Rautulwar's [`fake-review-detection`](https://github.com/Vinay-rautulwar/fake-review-detection) repository, commit `92c10b3`. It is a separate pipeline from the Ott hotel-corpus experiments in `scripts/` and `results/`.

## Expected data and limitations

The scripts expect a local 17-column CSV with review text, reviewer/product identifiers and metadata, and a `genuine`/`fake` label. The companion reports describe a 30,000-row template-generated dataset. That CSV and its split files are not included; see [`data/README.md`](../data/README.md) for the expected schema. Do not treat the companion dataset as naturally collected platform fraud data.

## Reproduce the data preparation and baseline

After obtaining a dataset whose provenance and redistribution terms are documented, place it at `data/raw/fake_review_dataset.csv`, then run:

```bash
pip install -r training/requirements-training.txt
python training/scripts/clean_dataset.py --raw_path data/raw/fake_review_dataset.csv
python training/scripts/validate_dataset.py --data_path data/processed/fake_review_dataset_clean.csv
python training/scripts/create_splits.py --input_path data/processed/fake_review_dataset_clean.csv --near_dup_path data/quarantine/near_duplicate_reviews.csv --seed 42
python training/scripts/train_baseline.py
```

The split script targets reviewer-held-out partitions and keeps detected near-duplicate clusters within a partition. It does not eliminate every possible source of leakage: the companion split report notes product overlap and potential lookahead in product-level aggregates. The text-only baseline avoids those metadata fields.

## DeBERTa notebook and API artifacts

`notebooks/train_deberta.ipynb` trains a text-only DeBERTa-v3 model using training and validation data; `scripts/evaluate_deberta_test.py` expects an untouched `data/splits/test.csv` and a complete local checkpoint/tokenizer. The published source snapshot does not include the dataset, split files, model weights, or tokenizer vocabulary. Its reported DeBERTa metrics are therefore not reproduced by this project. Multimodal metadata fusion remains a configuration/design scaffold, not a completed experiment.

Model weights and generated dataset CSVs are excluded from Git. The FastAPI service is a prototype and reports inference unavailable until compatible artifacts are supplied. See [`backend/README.md`](../backend/README.md) and [`reports/README.md`](../reports/README.md).
