#!/usr/bin/env python3
"""Evaluation script for video stabilization models."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from omegaconf import OmegaConf

from video_stabilization import VideoStabilizer
from video_stabilization.utils import MetricsCalculator, VideoProcessor, set_seed

logger = logging.getLogger(__name__)


def evaluate_method(
    method: str,
    input_video: Path,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single stabilization method.
    
    Args:
        method: Stabilization method name
        input_video: Path to input video
        config: Method configuration
        output_dir: Output directory
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating method: {method}")
    
    # Initialize stabilizer
    stabilizer = VideoStabilizer(method=method, config=config)
    
    # Load original video
    processor = VideoProcessor()
    original_frames = processor.load_video(input_video)
    
    # Stabilize video
    stabilized_frames = stabilizer.stabilize(input_video)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(original_frames, stabilized_frames)
    
    # Save stabilized video
    output_video_path = output_dir / f"{method}_stabilized.mp4"
    processor.save_video(stabilized_frames, output_video_path)
    
    # Add method name to metrics
    metrics["method"] = method
    metrics["input_video"] = str(input_video)
    metrics["output_video"] = str(output_video_path)
    
    logger.info(f"Completed evaluation for {method}")
    return metrics


def create_comparison_video(
    original_frames: torch.Tensor,
    stabilized_results: Dict[str, torch.Tensor],
    output_path: Path,
    fps: float = 30.0,
) -> None:
    """Create side-by-side comparison video.
    
    Args:
        original_frames: Original video frames
        stabilized_results: Dictionary of method -> stabilized frames
        output_path: Path to save comparison video
        fps: Frames per second
    """
    import cv2
    import numpy as np
    
    # Calculate grid dimensions
    num_methods = len(stabilized_results)
    cols = min(3, num_methods + 1)  # +1 for original
    rows = (num_methods + 1 + cols - 1) // cols
    
    # Get frame dimensions
    h, w = original_frames.shape[1:3]
    
    # Create comparison frames
    comparison_frames = []
    for i in range(len(original_frames)):
        # Create grid frame
        grid_frame = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        # Add original frame
        grid_frame[0:h, 0:w] = original_frames[i]
        
        # Add stabilized frames
        row, col = 0, 1
        for method, frames in stabilized_results.items():
            if col >= cols:
                row += 1
                col = 0
            
            grid_frame[row*h:(row+1)*h, col*w:(col+1)*w] = frames[i]
            
            # Add method label
            cv2.putText(
                grid_frame,
                method,
                (col*w + 10, row*h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            col += 1
        
        comparison_frames.append(grid_frame)
    
    # Save comparison video
    processor = VideoProcessor()
    processor.save_video(np.array(comparison_frames), output_path, fps)


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate video stabilization methods")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Config file path")
    parser.add_argument("--input-video", type=str, help="Input video path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override with command line args
    if args.input_video:
        config.input_video = args.input_video
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Set random seed
    set_seed(42)
    
    # Setup paths
    input_video = Path(config.input_video)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        return
    
    # Load original video for comparison
    processor = VideoProcessor()
    original_frames = processor.load_video(input_video)
    
    # Evaluate each method
    all_metrics = []
    stabilized_results = {}
    
    for method in config.methods:
        method_config = OmegaConf.load(f"configs/{method}.yaml")
        metrics = evaluate_method(method, input_video, method_config, output_dir)
        all_metrics.append(metrics)
        
        # Load stabilized frames for comparison
        stabilized_video_path = output_dir / f"{method}_stabilized.mp4"
        stabilized_frames = processor.load_video(stabilized_video_path)
        stabilized_results[method] = stabilized_frames
    
    # Create comparison video
    if config.save_comparison_video:
        comparison_path = output_dir / "comparison.mp4"
        create_comparison_video(original_frames, stabilized_results, comparison_path)
        logger.info(f"Saved comparison video to: {comparison_path}")
    
    # Save metrics to CSV
    if config.save_metrics_csv:
        df = pd.DataFrame(all_metrics)
        metrics_path = output_dir / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to: {metrics_path}")
        
        # Print summary
        print("\nEvaluation Results:")
        print("=" * 50)
        for _, row in df.iterrows():
            print(f"\n{row['method'].upper()}:")
            print(f"  PSNR: {row['psnr']:.2f}")
            print(f"  SSIM: {row['ssim']:.4f}")
            print(f"  Temporal Consistency: {row['temporal_consistency_stabilized']:.4f}")
            print(f"  Temporal Improvement: {row['temporal_improvement']:.4f}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
