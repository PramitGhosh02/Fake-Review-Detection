"""
Setup script for FakeGuard package.

FakeGuard is a multi-view deep learning framework with explainable attention
for fake review detection across e-commerce and review platforms.
"""

from pathlib import Path
from setuptools import find_packages, setup

# Read the long description from README.md
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="fakeguard",
    version="0.1.0",
    description="A Multi-View Deep Learning Framework with Explainable Attention for Fake Review Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FakeGuard Research Team",
    author_email="research@fakeguard.ai",
    url="https://github.com/fakeguard/fakeguard",
    package_dir={"": "."},
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
        "explain": [
            "shap>=0.42.0",
            "captum>=0.6.0",
        ],
        "graph": [
            "torch-geometric>=2.3.0",
            "torch-scatter",
            "torch-sparse",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
