"""
================================================================================
FakeGuard: Multi-View Deep Learning Framework for Fake Review Detection
================================================================================
File: notebooks/02_baseline_models.py
Description: Traditional Machine Learning Baselines for Fake Review Detection.
             Implements, trains, evaluates, and compares standard non-deep-learning
             baselines against the FakeGuard multi-view framework.

Baselines Implemented:
    1. Logistic Regression + TF-IDF (Text View Baseline)
    2. Support Vector Machine (Linear SVM) + TF-IDF (Text View Baseline)
    3. Naive Bayes (MultinomialNB) + TF-IDF (Text View Baseline)
    4. Random Forest + Behavioral Features (Behavioral View Baseline)
    5. Feature Fusion: Logistic Regression + (TF-IDF & Behavioral Features)

Outputs:
    - Results table saved to results/baseline_results.csv and .json
    - Publication LaTeX table saved to results/baseline_results.tex
    - Side-by-side Confusion Matrices: results/figures/baseline_confusion_matrices.png
    - ROC and PR Curves Comparison: results/figures/baseline_roc_curves.png, baseline_pr_curves.png
    - Metrics Comparison Bar Chart: results/figures/baseline_metrics_comparison.png
    - RF Feature Importance: results/figures/baseline_rf_feature_importance.png

Usage:
    - Run interactively in VS Code (using '# %%' cell markers) or Jupyter.
    - Run as a standalone Python script: python notebooks/02_baseline_models.py
================================================================================
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack, issparse

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

from src.data.feature_engineering import BehavioralFeatureExtractor
from src.data.preprocessing import TextPreprocessor
from src.evaluation.metrics import MetricsCalculator, ModelComparator
from data.download_data import generate_sample_dataset

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'semibold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Results Directory: {RESULTS_DIR}")

def load_dataset_for_modeling() -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads dataset and extracts text and behavioral features.
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Dataset and list of behavioral feature column names.
    """
    processed_file = PROCESSED_DIR / "dataset_with_features.csv"
    raw_file = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
    
    if processed_file.exists() and processed_file.stat().st_size > 100:
        print(f"Loading preprocessed dataset from: {processed_file}")
        df = pd.read_csv(processed_file)
    elif raw_file.exists() and raw_file.stat().st_size > 100:
        print(f"Extracting features from raw dataset: {raw_file}")
        df_raw = pd.read_csv(raw_file)
        extractor = BehavioralFeatureExtractor(df_raw)
        df = extractor.extract_all_features()
    else:
        print("No raw or processed dataset found. Generating synthetic dataset...")
        df_raw = generate_sample_dataset(str(PROJECT_ROOT / "data" / "raw"), num_reviews=3500)
        extractor = BehavioralFeatureExtractor(df_raw)
        df = extractor.extract_all_features()
        
    preprocessor = TextPreprocessor()
    if 'cleaned_text' not in df.columns:
        df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
        
    id_cols = {'review_id', 'user_id', 'product_id', 'label', 'text', 'cleaned_text', 'date', 'class_name'}
    behavioral_cols = [
        c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    for col in behavioral_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    return df, behavioral_cols

df, behavioral_features = load_dataset_for_modeling()
print(f"Dataset Loaded: {len(df)} samples | Fake: {(df['label'] == 1).sum()} | Genuine: {(df['label'] == 0).sum()}")
print(f"Behavioral Features ({len(behavioral_features)}): {behavioral_features}")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=df['label']
)

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"Train samples: {len(train_df)} (Fake: {(y_train==1).sum()}, Genuine: {(y_train==0).sum()})")
print(f"Test samples:  {len(test_df)} (Fake: {(y_test==1).sum()}, Genuine: {(y_test==0).sum()})")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train_df['cleaned_text'])
X_test_tfidf = tfidf.transform(test_df['cleaned_text'])

scaler = StandardScaler()
X_train_behav = scaler.fit_transform(train_df[behavioral_features])
X_test_behav = scaler.transform(test_df[behavioral_features])

X_train_combined = hstack([X_train_tfidf, X_train_behav]).tocsr()
X_test_combined = hstack([X_test_tfidf, X_test_behav]).tocsr()

print(f"TF-IDF Matrix Shape:       {X_train_tfidf.shape}")
print(f"Behavioral Matrix Shape:   {X_train_behav.shape}")
print(f"Combined Matrix Shape:     {X_train_combined.shape}")

class BaselineExperiment:
    """
    Manages training, probability prediction, and evaluation for baseline models.
    """
    def __init__(self, name: str, model: Any, feature_type: str):
        self.name = name
        self.model = model
        self.feature_type = feature_type
        self.y_pred: Optional[np.ndarray] = None
        self.y_prob: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}
        
    def fit_and_evaluate(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray
    ) -> 'BaselineExperiment':
        """Trains the model and evaluates performance."""
        print(f"Training {self.name} ({self.feature_type})...")
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)
        
        if hasattr(self.model, "predict_proba"):
            self.y_prob = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X_test)
            self.y_prob = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
        else:
            self.y_prob = self.y_pred.astype(float)
            
        calc = MetricsCalculator(y_test, self.y_pred, self.y_prob)
        self.metrics = calc.compute_all()
        return self

experiments: Dict[str, BaselineExperiment] = {
    "Logistic Regression (TF-IDF)": BaselineExperiment(
        name="Logistic Regression (TF-IDF)",
        model=LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
        feature_type="text"
    ),
    
    "Linear SVM (TF-IDF)": BaselineExperiment(
        name="Linear SVM (TF-IDF)",
        model=CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', random_state=RANDOM_SEED, max_iter=2000),
            cv=3
        ),
        feature_type="text"
    ),
    
    "Naive Bayes (TF-IDF)": BaselineExperiment(
        name="Naive Bayes (TF-IDF)",
        model=MultinomialNB(alpha=0.5),
        feature_type="text"
    ),
    
    "Random Forest (Behavioral)": BaselineExperiment(
        name="Random Forest (Behavioral)",
        model=RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        feature_type="behavior"
    ),
    
    "Combined LR (Text + Behavior)": BaselineExperiment(
        name="Combined LR (Text + Behavior)",
        model=LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
        feature_type="combined"
    )
}

for name, exp in experiments.items():
    if exp.feature_type == "text":
        exp.fit_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test)
    elif exp.feature_type == "behavior":
        exp.fit_and_evaluate(X_train_behav, y_train, X_test_behav, y_test)
    elif exp.feature_type == "combined":
        exp.fit_and_evaluate(X_train_combined, y_train, X_test_combined, y_test)

print("\nAll baseline models trained successfully.")

print("=" * 80)
print("                       DETAILED CLASSIFICATION REPORTS                       ")
print("=" * 80)

for name, exp in experiments.items():
    print(f"\n--- Model: {name} ---")
    print(classification_report(y_test, exp.y_pred, target_names=['Genuine (0)', 'Fake (1)'], digits=4))

records = []
for name, exp in experiments.items():
    row = {"Model": name, "Feature View": exp.feature_type.capitalize()}
    row.update(exp.metrics)
    records.append(row)

results_df = pd.DataFrame(records).set_index("Model")
columns_order = ['Feature View', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_binary', 'auc_roc', 'auc_pr']
available_cols = [c for c in columns_order if c in results_df.columns]
results_df = results_df[available_cols]

print("\n" + "=" * 80)
print("                     BASELINE PERFORMANCE COMPARISON TABLE                    ")
print("=" * 80)
display(results_df.round(4)) if 'display' in globals() else print(results_df.round(4).to_string())

results_csv_path = RESULTS_DIR / "baseline_results.csv"
results_json_path = RESULTS_DIR / "baseline_results.json"
results_tex_path = RESULTS_DIR / "baseline_results.tex"

results_df.round(4).to_csv(results_csv_path)
results_df.round(4).to_json(results_json_path, orient="index", indent=2)

with open(results_tex_path, "w") as f:
    f.write(results_df.round(4).to_latex(caption="Baseline Model Performance Comparison on Fake Review Detection", label="tab:baselines"))

print(f"\nExported Results Table to:")
print(f"  CSV:   {results_csv_path}")
print(f"  JSON:  {results_json_path}")
print(f"  LaTeX: {results_tex_path}")

num_models = len(experiments)
cols = 3
rows = (num_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes_flat = axes.flatten()

for idx, (name, exp) in enumerate(experiments.items()):
    ax = axes_flat[idx]
    cm = confusion_matrix(y_test, exp.y_pred)
    
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.array([[f"{count:,}\n({pct:.1f}%)" for count, pct in zip(row_count, row_pct)]
                      for row_count, row_pct in zip(cm, cm_pct)])
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap='Blues',
        cbar=False,
        ax=ax,
        xticklabels=['Genuine (0)', 'Fake (1)'],
        yticklabels=['Genuine (0)', 'Fake (1)'],
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    ax.set_title(f"{name}\nF1-Macro: {exp.metrics.get('f1_macro', 0):.3f} | AUC: {exp.metrics.get('auc_roc', 0):.3f}", pad=10)
    ax.set_ylabel('Ground Truth')
    ax.set_xlabel('Predicted Label')

for idx in range(num_models, len(axes_flat)):
    fig.delaxes(axes_flat[idx])

plt.suptitle('Baseline Models: Test Set Confusion Matrices', fontsize=16, y=1.02)
plt.tight_layout()

cm_fig_path = FIGURES_DIR / "baseline_confusion_matrices.png"
fig.savefig(cm_fig_path, dpi=300)
fig.savefig(FIGURES_DIR / "baseline_confusion_matrices.pdf")
print(f"Saved confusion matrices figure to: {cm_fig_path}")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for (name, exp), color in zip(experiments.items(), model_colors):
    if exp.y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, exp.y_prob)
        auc_val = roc_auc_score(y_test, exp.y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Chance (AUC = 0.500)')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curves')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
axes[0].set_ylabel('True Positive Rate (Recall / Sensitivity)')
axes[0].legend(loc='lower right', frameon=True)
axes[0].grid(True, linestyle='--', alpha=0.6)

no_skill = (y_test == 1).sum() / len(y_test)
axes[1].plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.7, label=f'Baseline Chance (AP = {no_skill:.3f})')

for (name, exp), color in zip(experiments.items(), model_colors):
    if exp.y_prob is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, exp.y_prob)
        ap_val = average_precision_score(y_test, exp.y_prob)
        axes[1].plot(recall_vals, precision_vals, label=f"{name} (AP = {ap_val:.3f})", color=color, linewidth=2)

axes[1].set_title('Precision-Recall (PR) Curves')
axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision (Positive Predictive Value)')
axes[1].legend(loc='upper right', frameon=True)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Discrimination & Precision-Recall Dynamics: Baseline Models', y=1.02)
plt.tight_layout()

roc_fig_path = FIGURES_DIR / "baseline_roc_curves.png"
pr_fig_path = FIGURES_DIR / "baseline_pr_curves.png"
fig.savefig(roc_fig_path, dpi=300)
print(f"Saved ROC/PR curves figure to: {roc_fig_path}")
plt.show()

plot_metrics = ['accuracy', 'f1_macro', 'auc_roc', 'auc_pr']
plot_df = results_df[plot_metrics].copy()
plot_df.columns = ['Accuracy', 'F1-Macro', 'AUC-ROC', 'AUC-PR']

ax = plot_df.plot(
    kind='bar',
    figsize=(12, 6.5),
    width=0.75,
    edgecolor='black',
    linewidth=0.8,
    colormap='viridis'
)

plt.title('Baseline Models Performance Across Key Evaluation Metrics', pad=15)
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=25, ha='right')
plt.ylim(0, 1.1)
plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)

plt.tight_layout()
metrics_bar_path = FIGURES_DIR / "baseline_metrics_comparison.png"
plt.savefig(metrics_bar_path, dpi=300)
print(f"Saved metrics comparison bar chart to: {metrics_bar_path}")
plt.show()

rf_exp = experiments.get("Random Forest (Behavioral)")
if rf_exp is not None and hasattr(rf_exp.model, "feature_importances_"):
    importances = rf_exp.model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': behavioral_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(behavioral_features) * 0.45)))
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color='#2b5c8f', edgecolor='black', linewidth=0.8, height=0.65)
    
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.set_title('Random Forest: Behavioral Feature Importances (Mean Decrease Impurity)', pad=15)
    ax.set_xlabel('Gini Importance')
    ax.set_ylabel('Behavioral Feature')
    ax.set_xlim(0, max(fi_df['Importance']) * 1.18)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    rf_fi_path = FIGURES_DIR / "baseline_rf_feature_importance.png"
    fig.savefig(rf_fi_path, dpi=300)
    print(f"Saved Random Forest feature importance figure to: {rf_fi_path}")
    plt.show()

lr_exp = experiments.get("Logistic Regression (TF-IDF)")
if lr_exp is not None and hasattr(lr_exp.model, "coef_"):
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = lr_exp.model.coef_[0]
    
    top_n = 10
    top_fake_idx = np.argsort(coefs)[-top_n:]
    top_gen_idx = np.argsort(coefs)[:top_n]
    
    top_lexical_df = pd.DataFrame({
        'Token': list(feature_names[top_fake_idx][::-1]) + list(feature_names[top_gen_idx]),
        'Coefficient': list(coefs[top_fake_idx][::-1]) + list(coefs[top_gen_idx]),
        'Association': ['Fake (Spam)'] * top_n + ['Genuine'] * top_n
    })
    
    print("\n" + "=" * 60)
    print("        TOP LEXICAL PREDICTORS (LOGISTIC REGRESSION)         ")
    print("=" * 60)
    print(top_lexical_df.to_string(index=False))

print("\n" + "=" * 80)
print("           TRADITIONAL ML BASELINE BENCHMARKING COMPLETE              ")
print(f"All figures saved to: {FIGURES_DIR}")
print(f"All metric tables saved to: {RESULTS_DIR}")
print("=" * 80)
