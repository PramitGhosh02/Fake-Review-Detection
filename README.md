# FakeGuard: A Multi-View Deep Learning Framework with Explainable Attention for Fake Review Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📌 Overview

**FakeGuard** is an end-to-end multi-view deep learning framework engineered for robust, explainable fake review detection across e-commerce and online review platforms (such as Yelp and Amazon). Online reviews heavily influence consumer decisions and platform trust, yet malicious actors frequently deploy deceptive reviews or organized spam campaigns to manipulate ratings.

Traditional detection approaches typically rely exclusively on linguistic content or isolated behavioral heuristics. In contrast, **FakeGuard** synergistically unifies three complementary dimensions:
1. **Linguistic View (Semantic Content):** Contextual embeddings from pre-trained language models (BERT/RoBERTa) that capture syntactic nuances, sentiment inconsistencies, and deceptive writing patterns.
2. **Behavioral View (Metadata & Temporal Dynamics):** Handcrafted and learned statistical representations capturing user rating deviations, review bursts, account velocity, and activity distributions.
3. **Relational View (Graph Structure):** Graph Neural Networks (GNNs via PyTorch Geometric) operating over heterogeneous reviewer-product-review interaction graphs to detect collusion rings and suspicious structural cliques.

These multi-modal representations are fused via an **Explainable Multi-Head Cross-Attention Fusion Mechanism**, providing interpretable attribution weights for platform administrators and consumers alike.

---

## 🏗️ Architecture Overview

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
                │  Multi-Head Cross-Attention       │
                │        Fusion Layer               │
                │    (Explainable Modality Weights) │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │   Classification Head (MLP)       │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │   Output: Genuine vs. Fake Review │
                │   + Multi-View Attention Heatmap  │
                └───────────────────────────────────┘
```

### Key Modules:
- **Text Encoder (`src/models/text_encoder.py`)**: Fine-tuned Transformer backbone extracting dense sentence representations.
- **Behavior Encoder (`src/models/behavior_encoder.py`)**: Multi-layer feed-forward network projecting numeric metadata into a shared latent space.
- **Graph Encoder (`src/models/graph_encoder.py`)**: Graph Attention Networks (GAT) learning node representations over heterogeneous reviewer-product bipartite graphs.
- **Explainable Fusion (`src/models/fusion.py`)**: Multi-head cross-attention mechanism dynamically weighting modality importance per sample.
- **Explainability Engine (`src/evaluation/explainability.py`)**: SHAP and Captum integrations for token-level, feature-level, and graph-subgraph attributions.

---

## 📂 Project Structure

```
PBL_Sem5/
├── configs/
│   └── config.yaml               # Master experimental configuration
├── data/
│   ├── raw/                      # Raw datasets (YelpChi, YelpZip, Amazon)
│   └── processed/                # Preprocessed splits, graphs, and features
├── notebooks/                    # Interactive analysis and prototyping
├── results/
│   ├── checkpoints/              # Serialized model weights
│   ├── logs/                     # TensorBoard / execution logs
│   └── figures/                  # ROC curves, confusion matrices, SHAP plots
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

## ⚡ Installation

### 1. Clone the repository & navigate to project directory:
```bash
git clone https://github.com/fakeguard/fakeguard.git
cd PBL_Sem5
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

## 🚀 Usage

### 1. Configuration
All hyperparameters, dataset paths, model architectures, and training settings are centralized in [`configs/config.yaml`](configs/config.yaml).

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
Run training using the standard configuration:
```bash
python -m src.training.train --config configs/config.yaml
```

### 3. Evaluation
Evaluate a trained model checkpoint on the test partition:
```bash
python -m src.evaluation.evaluate --config configs/config.yaml --checkpoint results/checkpoints/best_model.pt
```

### 4. Explainability & Analysis
Generate SHAP values, attention weight distributions, and modality contributions:
```bash
python -m src.evaluation.explain --checkpoint results/checkpoints/best_model.pt --sample-idx 0
```

---

## 📊 Supported Datasets

| Dataset | Domain | Reviews | Spammers / Fake Ratio | Graph Relations |
| :--- | :--- | :--- | :--- | :--- |
| **YelpChi** | Restaurant & Hotel | 67,395 | ~13.2% | User-Product-Review (UPR) |
| **YelpZip** | Restaurant & Hotel | 608,598 | ~13.1% | User-Product-Review (UPR) |
| **Amazon** | Musical Instruments / Electronics | Variable | Variable | Co-review / Co-purchasing |

---

## 🧪 Evaluation Metrics

FakeGuard evaluates classification performance across standard fraud detection metrics:
- **Accuracy & Balanced Accuracy**
- **Precision, Recall, F1-Score (Macro & Binary)**
- **Area Under ROC Curve (AUC-ROC)**
- **Area Under Precision-Recall Curve (AUC-PR)**
- **Inference Latency & Parameter Efficiency**

---

## 📖 Citation

If you use FakeGuard in your research, please cite:

```bibtex
@article{fakeguard2026,
  title={FakeGuard: A Multi-View Deep Learning Framework with Explainable Attention for Fake Review Detection},
  author={FakeGuard Research Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
