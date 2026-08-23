"""Command-line training script for FakeGuard.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --debug  # Small subset
"""
import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.utils.helpers import load_config, set_seed, get_device, setup_logging
# Real project imports (uncomment/implement when modules are available):
# from src.data.preprocessing import TextPreprocessor
# from src.data.feature_engineering import BehavioralFeatureExtractor
# from src.data.dataset import FakeReviewDataset, create_dataloaders
# from src.models.fakeguard import FakeGuard
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train FakeGuard model')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--debug', action='store_true', help='Use small subset for debugging')
    parser.add_argument('--model-type', type=str, default='fakeguard',
                       choices=['fakeguard', 'text_only', 'text_behavior', 'text_graph'],
                       help='Model variant for ablation studies')
    args = parser.parse_args()
    
    # 1. Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        # Fallback or exit
        logger.error(f"Failed to load config: {e}. Ensure utils.helpers is implemented correctly.")
        config = {'learning_rate': 2e-5, 'num_epochs': 5, 'save_dir': 'checkpoints'}
        
    # Setup logging globally
    log_dir = config.get('log_dir', 'logs')
    setup_logging(log_dir)
    
    # 2. Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # 3. Load and preprocess data
    # 4. Create datasets and dataloaders
    logger.info("Setting up data loaders...")
    try:
        from src.data.dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(config, debug=args.debug)
    except ImportError:
        logger.warning("Could not import dataloader logic. Mocking for now.")
        train_loader, val_loader = [], []
        
    # 5. Initialize model
    logger.info(f"Initializing model type: {args.model_type}")
    try:
        from src.models.fakeguard import FakeGuard
        model = FakeGuard(config)
    except ImportError:
        logger.warning("Could not import FakeGuard model. Please implement the model.")
        import torch.nn as nn
        model = nn.Linear(10, 2)
        
    # 6. Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # 7. Train
    logger.info("Starting training loop...")
    results = trainer.train()
    
    # 8. Print final results
    logger.info("Training completed.")
    best_metrics = results.get('best_metrics', {})
    logger.info(f"Best Validation Results: {best_metrics}")

if __name__ == '__main__':
    main()
