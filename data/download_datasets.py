"""
================================================================================
Download Real Datasets for FakeGuard
================================================================================
Downloads publicly available fake review detection datasets.

Usage:
    python data/download_datasets.py
================================================================================
"""
import os
import sys
import zipfile
import io
import pandas as pd
import requests
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def download_deceptive_opinion_spam():
    """
    Dataset 1: Deceptive Opinion Spam Corpus (Ott et al., 2011 & 2013)
    - 1,600 hotel reviews (800 truthful + 800 deceptive)
    - Balanced 50/50 split
    - Gold standard benchmark dataset
    - Source: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus
    """
    print("\n" + "="*60)
    print("  Dataset 1: Deceptive Opinion Spam Corpus")
    print("  (Ott et al., 2011 - 1,600 hotel reviews)")
    print("="*60)
    
    url = "https://raw.githubusercontent.com/SarahMohaworku/Fake-Review-Detection/main/deceptive-opinion.csv"
    output_path = DATASETS_DIR / "deceptive-opinion.csv"
    
    if output_path.exists():
        print(f"  Already downloaded: {output_path}")
        df = pd.read_csv(output_path)
        print(f"  Rows: {len(df)}")
        return df
    
    print(f"  Downloading from GitHub mirror...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        df = pd.read_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")
        print(f"  Label distribution:\n{df['deceptive'].value_counts().to_string()}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Manual download: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus")
        return None


def download_fake_reviews_40k():
    """
    Dataset mirror containing records labeled CG and OR.
    Verify the primary source and construction process before research use.
    CG/OR labels indicate the file's generation-source categories and do not
    by themselves establish malicious deception or verified authenticity.
    """
    print("\n" + "="*60)
    print("  Dataset 2: Fake Reviews Dataset (40K)")
    print("  (20K original + 20K AI-generated)")
    print("="*60)
    
    url = "https://raw.githubusercontent.com/SarahMohaworku/Fake-Review-Detection/main/fake%20reviews%20dataset.csv"
    output_path = DATASETS_DIR / "fake reviews dataset.csv"
    
    if output_path.exists():
        print(f"  Already downloaded: {output_path}")
        df = pd.read_csv(output_path)
        print(f"  Rows: {len(df)}")
        return df
    
    print(f"  Downloading from GitHub mirror...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        df = pd.read_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")
        if 'label' in df.columns:
            print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Manual download: https://www.kaggle.com/datasets/jillanisofttech/fake-reviews-dataset")
        return None


def download_yelp_labeled():
    """
    Attempt to retrieve a Yelp review dataset from third-party mirrors.
    Verify the source, subset, and meaning of its labels before calling it
    YelpChi or treating filtered/recommended labels as ground truth.
    """
    print("\n" + "="*60)
    print("  Dataset 3: Yelp Labeled Reviews")
    print("  (Restaurant & Hotel reviews with labels)")
    print("="*60)

    # Try multiple sources
    urls = [
        "https://raw.githubusercontent.com/SarahMohaworku/Fake-Review-Detection/main/yelp_labelled.csv",
        "https://raw.githubusercontent.com/shebuti/YelpChi/master/output/YelpChi.csv",
    ]
    
    output_path = DATASETS_DIR / "yelp_labeled.csv"
    
    if output_path.exists():
        print(f"  Already downloaded: {output_path}")
        df = pd.read_csv(output_path)
        print(f"  Rows: {len(df)}")
        return df
    
    for url in urls:
        print(f"  Trying: {url[:60]}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv(output_path)
            print(f"  Saved to: {output_path}")
            print(f"  Rows: {len(df)} | Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("  Could not auto-download. Verify any manual dataset's provenance and label meaning:")
    print("  1. https://www.kaggle.com/datasets/virenraina/yelp-fake-review-dataset")
    print("  2. Request from http://odds.cs.stonybrook.edu/yelpchi-dataset/")
    return None


def create_dataset_readme():
    """Preserve the curated, caveated inventory instead of replacing it with unverified claims."""
    readme_path = DATASETS_DIR / "README.md"
    if readme_path.exists():
        print(f"\n  Keeping dataset notes: {readme_path}")
        return

    readme_path.write_text(
        "# Dataset inventory\n\n"
        "Record each corpus's primary source, label meaning, collection process, and limits "
        "before using it. Do not equate platform-filtered labels, deceptive reviews, and "
        "machine-generated text. See the project research audit for the experiment protocol.\n",
        encoding="utf-8",
    )
    print(f"\n  README saved to: {readme_path}")


def main():
    print("="*60)
    print("  FakeGuard — Dataset Downloader")
    print("="*60)
    
    d1 = download_deceptive_opinion_spam()
    d2 = download_fake_reviews_40k()
    d3 = download_yelp_labeled()
    create_dataset_readme()
    
    print("\n" + "="*60)
    print("  DOWNLOAD SUMMARY")
    print("="*60)
    
    datasets = [
        ("Deceptive Opinion Spam (1.6K)", d1),
        ("Fake Reviews 40K", d2),
        ("Yelp Labeled", d3),
    ]
    
    for name, df in datasets:
        status = f"{len(df)} rows" if df is not None else "FAILED - download manually"
        print(f"  {name}: {status}")
    
    print(f"\n  All datasets saved to: {DATASETS_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
