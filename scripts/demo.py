#!/usr/bin/env python3
"""Simple demo script for video stabilization."""

import argparse
import logging
from pathlib import Path

from video_stabilization import VideoStabilizer
from video_stabilization.utils import VideoProcessor, create_synthetic_shaky_video, set_seed

logger = logging.getLogger(__name__)


def main() -> None:
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Video stabilization demo")
    parser.add_argument("--method", type=str, default="optical_flow", 
                       choices=["optical_flow", "homography", "deep"],
                       help="Stabilization method")
    parser.add_argument("--input-video", type=str, help="Input video path")
    parser.add_argument("--output-video", type=str, help="Output video path")
    parser.add_argument("--create-synthetic", action="store_true", 
                       help="Create synthetic shaky video for demo")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set random seed
    set_seed(42)
    
    # Create synthetic video if requested
    if args.create_synthetic:
        input_video = "demo_input.mp4"
        create_synthetic_shaky_video(input_video, num_frames=100, shake_intensity=15.0)
        logger.info(f"Created synthetic shaky video: {input_video}")
    else:
        input_video = args.input_video
        if not input_video or not Path(input_video).exists():
            logger.error("Please provide a valid input video or use --create-synthetic")
            return
    
    # Set output video path
    output_video = args.output_video or f"demo_output_{args.method}.mp4"
    
    # Initialize stabilizer
    logger.info(f"Initializing {args.method} stabilizer...")
    stabilizer = VideoStabilizer(method=args.method)
    
    # Stabilize video
    logger.info("Stabilizing video...")
    stabilized_frames = stabilizer.stabilize(input_video, output_video)
    
    logger.info(f"Demo completed! Stabilized video saved to: {output_video}")
    
    # Calculate and display metrics
    processor = VideoProcessor()
    original_frames = processor.load_video(input_video)
    
    from video_stabilization.utils import MetricsCalculator
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(original_frames, stabilized_frames)
    
    print("\nStabilization Results:")
    print("=" * 30)
    print(f"Method: {args.method}")
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"Temporal Consistency Improvement: {metrics['temporal_improvement']:.4f}")


if __name__ == "__main__":
    main()
