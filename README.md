# FakeGuard: A Proposed Multi-View Deep Learning Framework for Fake Review Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

##  Overview

**FakeGuard** is a research prototype for studying text, behavioral, and relational signals in review fraud detection. It has not been validated as a robust real-world detector. The saved classical baseline results use a synthetic development dataset, and there are no valid test results for the complete three-view model.

Prior work already combines review metadata and network structure. FakeGuard's proposed design studies three complementary views:
1. **Linguistic View (Semantic Content):** Contextual embeddings from pre-trained language models (BERT/RoBERTa) that capture syntactic nuances, sentiment inconsistencies, and deceptive writing patterns.
2. **Behavioral View (Metadata & Temporal Dynamics):** Handcrafted and learned statistical representations capturing user rating deviations, review bursts, account velocity, and activity distributions.
3. **Proposed Relational View (Graph Structure):** A graph model over reviewer-review-product relations, if verified entity IDs and time-respecting edges are available. Current training does not construct these inputs.

The proposed design fuses these views with multi-head self-attention over a three-token view sequence. The attention matrix is a diagnostic, not a verified explanation. Current training does not supply graph inputs, so the graph embedding is zero in those runs.

### Research status

- The saved baseline table is reproducible on 2,000 synthetic records (315 labeled fake), with a stratified 80/20 split. It is a development check, not evidence of performance on real reviews.
- Separate text-only experiments have now been run on the local deceptive-opinion hotel corpus after exact-text deduplication. The best hotel-disjoint mean macro-F1 was 0.8859 for TF-IDF + linear SVM; a randomly initialized BiLSTM reached 0.7526. These are controlled-corpus results, not validation of the full multi-view model or performance on platform fraud. Reproduce them with `python scripts/run_ott_experiments.py` and `python scripts/run_ott_bilstm.py`.
- The Ott corpus and the machine-generated review corpus have no reviewer history or review graph. The repository's adapters synthesize label-correlated behavioral features for these datasets; those features are invalid for model evaluation.
- The local `new_data_test.csv` contains 2,483 rows, but its sampling and source are undocumented. It is not verified as the canonical YelpChi dataset.
- The evaluation and figure scripts include random or hard-coded values. Do not use their outputs as experimental results.
- See [the research audit](RESEARCH_AUDIT.md) for verified findings, sources, and the next experiment plan.

---

##  Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │              Input Review               │
                      └────────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│   Text Stream    │             │ Behavioral Stream│             │   Graph Stream   │
│  (Review Text)   │             │ (User/Item Meta) │             │ (Reviewer Graph) │
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                 │                                 │
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│   BERT/RoBERTa   │             │     Deep MLP     │             │     GAT / GCN    │
│  Text Encoder    │             │ Behavior Encoder │             │  Graph Encoder   │
│   (Dim: 768)     │             │   (Dim: 128)     │             │   (Dim: 128)     │
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                 │                                 │
         └────────────────────────┬────────┴─────────────────────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │  Multi-Head Self-Attention        │
                │        Fusion Layer               │
                │    (View Interaction Weights)     │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │   Classification Head (MLP)       │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │   Output: Dataset-Specific Label  │
                │   + Attention Diagnostic Matrix   │
                └───────────────────────────────────┘
```

### Key Modules:
- **Text Encoder (`src/models/text_encoder.py`)**: Fine-tuned Transformer backbone extracting dense sentence representations.
- **Behavior Encoder (`src/models/behavior_encoder.py`)**: Multi-layer feed-forward network projecting numeric metadata into a shared latent space.
- **Graph Encoder (`src/models/graph_encoder.py`)**: Homogeneous GAT layers. Graph construction and graph inputs are not connected to the main training entry point.
- **Fusion (`src/models/fusion.py`)**: Multi-head self-attention over the three projected view embeddings.
- **Explainability utilities (`src/evaluation/explainability.py`)**: A SHAP wrapper is present, but attention plots in this module are partly random or hard-coded and no valid explanations are reported.

---

##  Project Structure

```
PBL_Sem5/
├── configs/
│   └── config.yaml               # Master experimental configuration
├── data/
│   ├── raw/                      # Synthetic development dataset
│   ├── datasets/                 # Local review corpora; provenance varies
│   └── processed/                # Derived features and adapted datasets
├── notebooks/                    # Interactive analysis and prototyping
├── results/
│   ├── checkpoints/              # Optional training output; no validated checkpoint is included
│   ├── logs/                     # Optional training output
│   └── figures/                  # Development plots; verify provenance before use
├── src/
│   ├── __init__.py
│   ├── data/                     # Data loading, tokenization, graph builders
│   │   └── __init__.py
│   ├── models/                   # Neural network architectures & encoders
│   │   └── __init__.py
│   ├── training/                 # Training engine, optimizers, early stopping
│   │   └── __init__.py
│   ├── evaluation/               # Metrics calculation and explainability tools
│   │   └── __init__.py
│   └── utils/                    # Seed setup, logging, device handling, checkpoints
│       ├── __init__.py
│       └── helpers.py
├── tests/                        # Automated unit and integration tests
├── .gitignore
├── requirements.txt              # Environment dependencies
├── setup.py                      # Package installation script
└── README.md                     # Project documentation
```

---

##  Installation

### 1. Navigate to the project directory:
```bash
git clone https://github.com/PramitGhosh02/Fake-Review-Detection.git
cd Fake-Review-Detection
```

### 2. Create and activate a Python virtual environment:
```bash
# Using conda
conda create -n fakeguard python=3.10 -y
conda activate fakeguard

# Or using venv
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install core dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install the package in editable mode:
```bash
pip install -e .
```

---

##  Usage

### 1. Configuration
The configuration file contains defaults for the prototype. It does not establish that the listed dataset or hardware was used in a completed experiment.

```yaml
data:
  dataset: "yelp_chi"
  batch_size: 32
  max_seq_length: 256

model:
  text_encoder:
    name: "bert-base-uncased"
  fusion:
    projected_dim: 128
    num_heads: 4
```

### 2. Training
For a development smoke run on the local data (not a research result):
```bash
python scripts/train.py --debug --model-type text_only
```

The current `scripts/evaluate.py` generates random labels and scores; it does not load the checkpoint for evaluation. Do not use it for research reporting. The explainability helpers also need correction and validation before their plots are interpreted.

### Reproducing the research experiments

Raw review datasets are not stored in this repository. Review `data/datasets/README.md` for label definitions and limitations, then obtain the permitted dataset files and place them in `data/datasets/`. The experiment scripts expect `deceptive-opinion.csv` for the Ott hotel corpus and `fake reviews dataset.csv` for the CG/OR corpus. The downloader can retrieve the former from a public mirror; verify the provenance and terms of any downloaded corpus before use.

```bash
python data/download_datasets.py
python scripts/run_ott_experiments.py
python scripts/run_ott_bilstm.py
python scripts/run_salminen40k_experiments.py
python scripts/run_salminen40k_mlp.py
python scripts/tune_grouped_svm.py
```

The scripts save fold-level metrics, summaries, and run metadata under `results/`. Row-level split manifests and raw datasets are excluded from version control; the manifests can be regenerated from the local corpora.

---

##  Local datasets (not included in the repository)

| Expected local file | Rows / labels in the audited local copy | Correct use and limitation |
| :--- | :--- | :--- |
| `data/datasets/deceptive-opinion.csv` | 1,600; 800 truthful and 800 deceptive | Controlled, text-only deception benchmark; no behavior or graph. |
| `data/datasets/fake reviews dataset.csv` | 40,432; 20,216 CG and 20,216 OR | Source and label construction are undocumented here. CG/OR means generation source, not necessarily deceptive intent. |
| `data/datasets/new_data_test.csv` | 2,483; 1,271 flagged and 1,212 unflagged | Provenance and sampling are undocumented; not verified as full YelpChi. |
| `data/raw/dataset.csv` | 2,000 synthetic; 315 fake | Development only; labels influence rating and text-length generation. |

These counts describe local files used during the audit; the CSV files are not committed. Do not treat download mirrors as authoritative sources without checking their provenance and redistribution terms.

The canonical YelpChi counts commonly reported from Rayana and Akoglu are external dataset statistics. This repository does not contain that complete 67,395-review dataset. Its filtered/recommended labels are proxy labels, not independently verified fake/genuine truth.

---

##  Evaluation Metrics

The project intends to report standard fraud detection metrics after the evaluation pipeline is corrected:
- **Accuracy & Balanced Accuracy**
- **Precision, Recall, F1-Score (Macro & Binary)**
- **Area Under ROC Curve (AUC-ROC)**
- **Average Precision (AP)** (the current code's `auc_pr` field)
- **Inference Latency & Parameter Efficiency**

---


