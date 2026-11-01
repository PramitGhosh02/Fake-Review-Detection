"""
Module for downloading or generating the dataset for FakeGuard.
"""

import os
import random
import datetime
from pathlib import Path

import pandas as pd
import numpy as np


def generate_sample_dataset(output_dir: str, num_reviews: int = 2000) -> pd.DataFrame:
    """
    Generates a synthetic dataset for testing/development with realistic patterns.
    
    Args:
        output_dir (str): Directory to save the generated dataset.
        num_reviews (int): Number of synthetic reviews to generate.
        
    Returns:
        pd.DataFrame: The generated synthetic dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_users = int(num_reviews * 0.3)
    num_products = int(num_reviews * 0.1)
    
    users = [f"user_{i}" for i in range(num_users)]
    products = [f"prod_{i}" for i in range(num_products)]
    
    start_date = datetime.datetime(2023, 1, 1)
    
    data = []
    for i in range(num_reviews):
        user_id = random.choice(users)
        product_id = random.choice(products)
        
        label = 1 if random.random() < 0.15 else 0
        
        if label == 1:
            rating = random.choice([1.0, 5.0])
            text_length = random.randint(10, 50)
        else:
            rating = float(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3]))
            text_length = random.randint(20, 150)
            
        words = ["great", "terrible", "good", "bad", "awesome", "awful", "the", "a", "is", "was", "product", "item", "service", "horrible", "fantastic"]
        text = " ".join(random.choices(words, k=text_length))
        
        days_offset = random.randint(0, 365)
        review_date = start_date + datetime.timedelta(days=days_offset)
        
        data.append({
            "review_id": f"rev_{i}",
            "user_id": user_id,
            "product_id": product_id,
            "rating": rating,
            "text": text,
            "label": label,
            "date": review_date.strftime("%Y-%m-%d")
        })
        
    df = pd.DataFrame(data)
    
    output_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} reviews at {output_path}")
    
    return df

def download_yelpchi(output_dir: str) -> None:
    """
    Attempts to download the YelpCHI dataset.
    Falls back to generating synthetic data if download fails or URL is unavailable.
    """
    print("YelpCHI dataset requires manual request from the authors (http://netspam.lab.uic.edu/).")
    print("Falling back to generating a synthetic dataset for development.")
    generate_sample_dataset(output_dir)

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, "raw")
    
    download_yelpchi(raw_dir)
