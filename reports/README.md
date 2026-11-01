# Reports and provenance

This folder contains reports imported from Vinay Rautulwar's [`fake-review-detection`](https://github.com/Vinay-rautulwar/fake-review-detection) repository at commit `92c10b3`. They document a separate, template-generated 30,000-row dataset and are not the results of the Ott deceptive-opinion experiments reported in `paper/main.tex`.

## Included source reports

- `dataset_audit.txt` and `split_report.txt`: collaborator-reported data quality and reviewer/near-duplicate split diagnostics. The source data and split CSVs are not included here.
- `baseline_validation_report.txt`: a reported validation-set TF-IDF + LightGBM baseline, not a test result.
- `deberta_validation_report.txt` and `deberta_test_evaluation_report.txt`: collaborator reports describing DeBERTa training and a later held-out test evaluation.

The DeBERTa test report claims accuracy 0.7143, fake-class precision 0.8674, recall 0.2809, F1 0.4243, ROC-AUC 0.7152, and average precision 0.6639. These values are not independently reproducible from this repository snapshot: `data/splits/test.csv`, the source dataset, `model.safetensors`, and tokenizer vocabulary are absent. Treat them as collaborator-reported preliminary figures only. Do not combine them into the same comparison table as the Ott corpus or imply that the baseline validation scores are a same-split test comparison.

The row-level cleaning log and raw dataset/readiness output containing review excerpts were not copied. Generated CSV datasets, split files, and model weights remain excluded by `.gitignore`.
