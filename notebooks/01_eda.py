"""
================================================================================
FakeGuard: Multi-View Deep Learning Framework for Fake Review Detection
================================================================================
File: notebooks/01_eda.py
Description: Exploratory Data Analysis (EDA) for the FakeGuard dataset.
             Analyzes class distribution, review text characteristics,
             rating distributions, temporal behavior, word clouds,
             and behavioral feature correlations.

Usage:
    - Run interactively in VS Code (using '# %%' cell markers) or Jupyter.
    - Run as a standalone Python script: python notebooks/01_eda.py
================================================================================
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

from src.data.feature_engineering import BehavioralFeatureExtractor
from src.data.preprocessing import TextPreprocessor
from data.download_data import generate_sample_dataset

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'semibold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'figure.titlesize': 15,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.autolayout': False,
})

PALETTE = {
    "Genuine": "#2b5c8f",
    "Fake": "#d95f02",
    "Neutral": "#7570b3",
    "Highlight": "#1b9e77"
}
CLASS_COLORS = [PALETTE["Genuine"], PALETTE["Fake"]]

print(f"Project Root: {PROJECT_ROOT}")
print(f"Figures Output Directory: {FIGURES_DIR}")

def load_or_generate_dataset(data_dir: Path, num_synthetic_samples: int = 3500) -> pd.DataFrame:
    """
    Loads raw review data or generates a representative synthetic dataset.
    
    Args:
        data_dir: Directory containing raw data files.
        num_synthetic_samples: Number of samples if generation is required.
        
    Returns:
        pd.DataFrame: Cleaned baseline dataset.
    """
    raw_csv = data_dir / "dataset.csv"
    
    if raw_csv.exists() and raw_csv.stat().st_size > 100:
        print(f"Loading existing raw dataset from: {raw_csv}")
        df = pd.read_csv(raw_csv)
    else:
        print(f"No raw dataset found at {raw_csv}. Generating {num_synthetic_samples} synthetic reviews...")
        df = generate_sample_dataset(output_dir=str(data_dir), num_reviews=num_synthetic_samples)
        
    expected_cols = ['review_id', 'user_id', 'product_id', 'rating', 'text', 'label', 'date']
    for col in expected_cols:
        if col not in df.columns:
            if col == 'label':
                df['label'] = 0
            elif col == 'date':
                df['date'] = datetime.date.today().strftime("%Y-%m-%d")
            elif col == 'rating':
                df['rating'] = 5.0
            elif col == 'text':
                df['text'] = ""
                
    df['label'] = df['label'].astype(int)
    df['rating'] = df['rating'].astype(float)
    df['text'] = df['text'].fillna('').astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    return df

raw_data_dir = PROJECT_ROOT / "data" / "raw"
df_raw = load_or_generate_dataset(raw_data_dir)

print(f"Dataset successfully loaded. Shape: {df_raw.shape}")
df_raw.head()

def compute_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches dataset with lexical and structural text characteristics.
    """
    df = df.copy()
    preprocessor = TextPreprocessor()
    
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    df['char_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df.apply(
        lambda row: (row['char_length'] / max(row['word_count'], 1)), axis=1
    )
    df['uppercase_count'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['uppercase_ratio'] = df.apply(
        lambda row: row['uppercase_count'] / max(row['char_length'], 1), axis=1
    )
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['question_count'] = df['text'].apply(lambda x: x.count('?'))
    df['class_name'] = df['label'].map({0: 'Genuine', 1: 'Fake'})
    return df

df_stats = compute_text_statistics(df_raw)

total_reviews = len(df_stats)
num_users = df_stats['user_id'].nunique()
num_products = df_stats['product_id'].nunique()
fake_count = (df_stats['label'] == 1).sum()
genuine_count = (df_stats['label'] == 0).sum()
fake_ratio = (fake_count / total_reviews) * 100
genuine_ratio = (genuine_count / total_reviews) * 100

print("=" * 60)
print("             FAKEMGUARD DATASET OVERVIEW SUMMARY            ")
print("=" * 60)
print(f"Total Reviews:          {total_reviews:,}")
print(f"Unique Users:           {num_users:,} (Avg {total_reviews/max(num_users, 1):.2f} reviews/user)")
print(f"Unique Products:        {num_products:,} (Avg {total_reviews/max(num_products, 1):.2f} reviews/product)")
print(f"Date Range:             {df_stats['date'].min().strftime('%Y-%m-%d')} to {df_stats['date'].max().strftime('%Y-%m-%d')}")
print(f"Genuine Reviews (0):    {genuine_count:,} ({genuine_ratio:.2f}%)")
print(f"Fake Reviews (1):       {fake_count:,} ({fake_ratio:.2f}%)")
print(f"Class Imbalance Ratio:  1 : {genuine_count / max(fake_count, 1):.2f}")
print("=" * 60)

summary_cols = ['word_count', 'char_length', 'avg_word_length', 'uppercase_ratio', 'exclamation_count', 'rating']
class_summary = df_stats.groupby('class_name')[summary_cols].agg(['mean', 'std', 'median', 'min', 'max'])
display(class_summary) if 'display' in globals() else print(class_summary.to_string())

fig, ax = plt.subplots(figsize=(7, 5))

counts = [genuine_count, fake_count]
labels = ['Genuine (0)', 'Fake (1)']
bars = ax.bar(labels, counts, color=CLASS_COLORS, width=0.55, edgecolor='black', linewidth=1.2, alpha=0.9)

for bar, count, pct in zip(bars, counts, [genuine_ratio, fake_ratio]):
    height = bar.get_height()
    ax.annotate(
        f'{count:,}\n({pct:.1f}%)',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 6),
        textcoords="offset points",
        ha='center', va='bottom',
        fontsize=11, fontweight='bold'
    )

ax.set_title('Dataset Class Distribution (FakeGuard)', pad=15)
ax.set_ylabel('Number of Reviews')
ax.set_xlabel('Review Category')
ax.set_ylim(0, max(counts) * 1.18)
ax.grid(axis='y', linestyle='--', alpha=0.7)

class_dist_path = FIGURES_DIR / "eda_class_distribution.png"
fig.savefig(class_dist_path, dpi=300)
fig.savefig(FIGURES_DIR / "eda_class_distribution.pdf")
print(f"Saved class balance figure to: {class_dist_path}")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.histplot(
    data=df_stats,
    x='word_count',
    hue='class_name',
    palette={'Genuine': PALETTE['Genuine'], 'Fake': PALETTE['Fake']},
    kde=True,
    element='step',
    stat='density',
    common_norm=False,
    bins=30,
    ax=axes[0],
    alpha=0.4
)
axes[0].set_title('Review Word Count Distribution')
axes[0].set_xlabel('Word Count per Review')
axes[0].set_ylabel('Density')
axes[0].grid(True, linestyle='--', alpha=0.6)

sns.boxplot(
    data=df_stats,
    x='class_name',
    y='word_count',
    palette={'Genuine': PALETTE['Genuine'], 'Fake': PALETTE['Fake']},
    width=0.45,
    boxprops=dict(alpha=0.85),
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
    ax=axes[1]
)
axes[1].set_title('Review Length Comparison (Median & IQR)')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Word Count')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Lexical Length Characteristics: Genuine vs. Fake Reviews', y=1.02)
plt.tight_layout()

length_dist_path = FIGURES_DIR / "eda_review_length_distribution.png"
fig.savefig(length_dist_path, dpi=300)
print(f"Saved review length figure to: {length_dist_path}")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.countplot(
    data=df_stats,
    x='rating',
    hue='class_name',
    palette={'Genuine': PALETTE['Genuine'], 'Fake': PALETTE['Fake']},
    edgecolor='black',
    linewidth=0.8,
    ax=axes[0]
)
axes[0].set_title('Rating Counts by Class')
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].legend(title='Category')

rating_prop = (
    df_stats.groupby('class_name')['rating']
    .value_counts(normalize=True)
    .rename('proportion')
    .reset_index()
)

sns.barplot(
    data=rating_prop,
    x='rating',
    y='proportion',
    hue='class_name',
    palette={'Genuine': PALETTE['Genuine'], 'Fake': PALETTE['Fake']},
    edgecolor='black',
    linewidth=0.8,
    ax=axes[1]
)
axes[1].set_title('Normalized Rating Proportions (Polarization Check)')
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Proportion within Class')
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
axes[1].legend(title='Category')

plt.suptitle('Star Rating Patterns & Polarization Analysis', y=1.02)
plt.tight_layout()

rating_dist_path = FIGURES_DIR / "eda_rating_distribution.png"
fig.savefig(rating_dist_path, dpi=300)
print(f"Saved rating distribution figure to: {rating_dist_path}")
plt.show()

def generate_wordclouds(df: pd.DataFrame, save_path: Path):
    """
    Generates side-by-side word clouds for genuine and fake reviews.
    Falls back gracefully to term frequency bar plots if wordcloud is unavailable.
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
        has_wordcloud = True
    except ImportError:
        has_wordcloud = False
        print("Note: 'wordcloud' package not found. Generating top word frequency charts instead.")

    genuine_text = " ".join(df[df['label'] == 0]['cleaned_text'].tolist())
    fake_text = " ".join(df[df['label'] == 1]['cleaned_text'].tolist())
    
    custom_stopwords = set(['the', 'and', 'a', 'to', 'of', 'is', 'it', 'in', 'i', 'this', 'that', 'was', 'for', 'with', 'on'])

    if has_wordcloud and len(genuine_text.strip()) > 0 and len(fake_text.strip()) > 0:
        wc_stopwords = STOPWORDS.union(custom_stopwords)
        
        wc_gen = WordCloud(
            width=800, height=450,
            background_color='white',
            colormap='Blues',
            stopwords=wc_stopwords,
            max_words=100,
            random_state=42
        ).generate(genuine_text)

        wc_fake = WordCloud(
            width=800, height=450,
            background_color='white',
            colormap='YlOrRd',
            stopwords=wc_stopwords,
            max_words=100,
            random_state=42
        ).generate(fake_text)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(wc_gen, interpolation='bilinear')
        axes[0].set_title('Genuine Reviews Word Cloud', fontsize=14, fontweight='bold', pad=12)
        axes[0].axis('off')

        axes[1].imshow(wc_fake, interpolation='bilinear')
        axes[1].set_title('Fake Reviews Word Cloud', fontsize=14, fontweight='bold', pad=12)
        axes[1].axis('off')

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (lbl, name, col, ax) in enumerate([
            (0, 'Genuine', PALETTE['Genuine'], axes[0]),
            (1, 'Fake', PALETTE['Fake'], axes[1])
        ]):
            words = [w for text in df[df['label'] == lbl]['cleaned_text'] for w in text.split() if w not in custom_stopwords and len(w) > 2]
            top_words = pd.Series(words).value_counts().head(15)
            sns.barplot(x=top_words.values, y=top_words.index, color=col, ax=ax, edgecolor='black', linewidth=0.8)
            ax.set_title(f'Top Words in {name} Reviews')
            ax.set_xlabel('Frequency')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.show()

wordcloud_path = FIGURES_DIR / "eda_wordclouds.png"
generate_wordclouds(df_stats, wordcloud_path)
print(f"Saved word cloud / lexical chart to: {wordcloud_path}")

print("Extracting behavioral features across users and products...")
feature_extractor = BehavioralFeatureExtractor(df_raw)
df_features = feature_extractor.extract_all_features()

print(f"Feature extraction completed. Total columns: {df_features.shape[1]}")

id_cols = {'review_id', 'user_id', 'product_id', 'label', 'text', 'cleaned_text', 'date', 'class_name'}
behavioral_cols = [
    col for col in df_features.columns
    if col not in id_cols and pd.api.types.is_numeric_dtype(df_features[col])
]

print("Extracted Behavioral Features:")
for i, col in enumerate(behavioral_cols, 1):
    print(f"  {i:2d}. {col}")

feature_diffs = []
for col in behavioral_cols:
    gen_vals = df_features[df_features['label'] == 0][col].dropna()
    fake_vals = df_features[df_features['label'] == 1][col].dropna()
    
    mean_gen = gen_vals.mean()
    mean_fake = fake_vals.mean()
    pooled_std = np.sqrt((gen_vals.var() + fake_vals.var()) / 2.0) + 1e-9
    cohen_d = (mean_fake - mean_gen) / pooled_std
    
    feature_diffs.append({
        'feature': col,
        'mean_genuine': mean_gen,
        'mean_fake': mean_fake,
        'standardized_diff': cohen_d,
        'abs_diff': abs(cohen_d)
    })

df_diffs = pd.DataFrame(feature_diffs).sort_values(by='abs_diff', ascending=True)

fig, ax = plt.subplots(figsize=(9, max(5, len(behavioral_cols) * 0.45)))
bar_colors = [PALETTE['Fake'] if val > 0 else PALETTE['Genuine'] for val in df_diffs['standardized_diff']]

bars = ax.barh(
    df_diffs['feature'],
    df_diffs['standardized_diff'],
    color=bar_colors,
    edgecolor='black',
    linewidth=0.8,
    height=0.65
)

ax.axvline(0, color='black', linewidth=1.0, linestyle='-')
ax.set_title('Top Behavioral Differences: Genuine vs. Fake Reviews\n(Standardized Effect Size: [Fake - Genuine] / Pooled Std)', pad=15)
ax.set_xlabel('Standardized Difference (Cohen\'s d)')
ax.set_ylabel('Behavioral Feature')
ax.grid(axis='x', linestyle='--', alpha=0.6)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PALETTE['Fake'], edgecolor='black', label='Elevated in Fake Reviews'),
    Patch(facecolor=PALETTE['Genuine'], edgecolor='black', label='Elevated in Genuine Reviews')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
top_features_path = FIGURES_DIR / "eda_top_features_comparison.png"
fig.savefig(top_features_path, dpi=300)
print(f"Saved top features comparison figure to: {top_features_path}")
plt.show()

corr_matrix = df_features[behavioral_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    vmax=1.0,
    vmin=-1.0,
    center=0,
    annot=True,
    fmt='.2f',
    square=True,
    linewidths=0.7,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
    ax=ax
)

ax.set_title('Behavioral Features Correlation Matrix (FakeGuard)', pad=18)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_path = FIGURES_DIR / "eda_behavioral_correlation_heatmap.png"
fig.savefig(heatmap_path, dpi=300)
fig.savefig(FIGURES_DIR / "eda_behavioral_correlation_heatmap.pdf")
print(f"Saved correlation heatmap to: {heatmap_path}")
plt.show()

output_processed_csv = PROCESSED_DIR / "dataset_with_features.csv"
df_features.to_csv(output_processed_csv, index=False)
print(f"Successfully exported feature-enriched dataset to: {output_processed_csv}")
print("Exploratory Data Analysis completed successfully.")
