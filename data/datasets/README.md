# Dataset inventory and limits

This file documents what is present in this repository. The corpora represent different tasks and label constructions; do not merge them into one binary fake/genuine dataset without an explicit, justified mapping.

## `deceptive-opinion.csv`

- Local size: 1,600 hotel reviews; labels are balanced (800 truthful, 800 deceptive).
- Source: Ott et al. (2011), *Finding Deceptive Opinion Spam by Any Stretch of the Imagination*, ACL 2011. [ACL Anthology record](https://aclanthology.org/P11-1032/)
- Task: controlled text deception classification.
- Limitation: the corpus does not supply reviewer histories, timestamps, or an interaction graph. Use text-only experiments. Do not add synthetic behavioral features.

## `fake reviews dataset.csv`

- Local size: 40,432 rows; labels are balanced (20,216 `CG`, 20,216 `OR`).
- The CSV's labels identify `CG` and `OR`, but source, collection process, and label construction are not documented in this repository.
- Do not interpret `CG` as malicious deception or `OR` as verified authentic reviews until the primary source and labeling method are checked. Use a separate machine-generated-versus-other task if provenance is confirmed.
- The local file does not provide real reviewer histories or graph relations. Do not use generated proxy behavior as model evidence.

## `new_data_test.csv`

- Local size: 2,483 rows; 1,271 `flagged`, 1,212 unflagged.
- It contains fields resembling Yelp review metadata, but the source, sample-selection procedure, and exact meaning of `flagged` are not documented here.
- Do not call this the complete or canonical YelpChi dataset until provenance is established.

## Canonical YelpChi reference (not present as a full local dataset)

Rayana and Akoglu report 67,395 reviews, 38,063 users, and 201 businesses for YelpChi. The labels refer to Yelp-filtered versus recommended reviews. These are platform-filter proxy labels, not independently verified ground-truth fraud annotations. See the original KDD paper: [Collective Opinion Spam Detection: Bridging Review Networks and Metadata](https://doi.org/10.1145/2783258.2783370).

## Synthetic development data

`../raw/dataset.csv` contains 2,000 synthetic records. The generator samples labels first and then uses label-dependent rating and text-length distributions. It is suitable for pipeline smoke checks only, never for performance claims or research conclusions.

## Recommended research practice

Keep these tasks separate, record source/version and label semantics, use only features available at prediction time, split before fitting scalers or computing aggregate behavior, and build graph snapshots without future test data. The full project assessment and experiment plan are in [`../../RESEARCH_AUDIT.md`](../../RESEARCH_AUDIT.md).
