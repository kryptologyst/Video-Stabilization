#!/usr/bin/env python3
"""Training script for video stabilization models."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from video_stabilization.models import DeepStabilizer
from video_stabilization.utils import set_seed, get_device

logger = logging.getLogger(__name__)


def create_synthetic_dataset(num_videos: int = 100, output_dir: Path = Path("data/synthetic")) -> None:
    """Create synthetic dataset for training.
    
    Args:
        num_videos: Number of synthetic videos to create
        output_dir: Directory to save synthetic videos
    """
    from video_stabilization.utils import create_synthetic_shaky_video
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_videos):
        video_path = output_dir / f"shaky_video_{i:04d}.mp4"
        create_synthetic_shaky_video(
            video_path,
            num_frames=50,
            shake_intensity=5.0 + i * 0.1
        )
    
    logger.info(f"Created {num_videos} synthetic videos in {output_dir}")


def train_model(config: Dict[str, Any]) -> None:
    """Train the deep stabilization model.
    
    Args:
        config: Training configuration
    """
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Get device
    device = get_device(config.get("device"))
    logger.info(f"Using device: {device}")
    
    # Create synthetic dataset
    data_dir = Path(config.get("data_dir", "data/synthetic"))
    if not data_dir.exists():
        create_synthetic_dataset(
            num_videos=config.get("num_synthetic_videos", 100),
            output_dir=data_dir
        )
    
    # Initialize model
    model_config = {
        "input_channels": config.get("input_channels", 6),
        "hidden_dim": config.get("hidden_dim", 256),
    }
    
    stabilizer = DeepStabilizer(model_config, device)
    model = stabilizer.model
    
    # Setup training
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-4)
    )
    
    criterion = nn.MSELoss()
    
    # Training loop (simplified - in practice you'd load real video data)
    model.train()
    for epoch in range(config.get("num_epochs", 100)):
        # This is a placeholder - in practice you'd load video pairs
        # and train on actual stabilization tasks
        optimizer.zero_grad()
        
        # Dummy loss for demonstration
        dummy_input = torch.randn(1, 6, 64, 64).to(device)
        dummy_target = torch.randn(1, 2).to(device)
        
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Save model
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "deep_stabilizer.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, checkpoint_path)
    
    logger.info(f"Saved model checkpoint to: {checkpoint_path}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video stabilization model")
    parser.add_argument("--config", type=str, default="configs/deep.yaml", help="Config file path")
    parser.add_argument("--device", type=str, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override with command line args
    if args.device:
        config.device = args.device
    config.seed = args.seed
    
    logger.info("Starting training...")
    train_model(config)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
