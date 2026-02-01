"""Core video stabilization functionality."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from .models import DeepStabilizer, HomographyStabilizer, OpticalFlowStabilizer
from .utils import VideoProcessor

logger = logging.getLogger(__name__)


class VideoStabilizer:
    """Main video stabilizer class that orchestrates different stabilization methods.
    
    This class provides a unified interface for video stabilization using various
    algorithms including optical flow, homography estimation, and deep learning approaches.
    
    Args:
        method: Stabilization method to use ('optical_flow', 'homography', 'deep')
        config: Configuration dictionary or DictConfig
        device: Device to run on ('cuda', 'mps', 'cpu')
    """
    
    def __init__(
        self,
        method: str = "optical_flow",
        config: Optional[Union[Dict[str, Any], DictConfig]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the video stabilizer."""
        self.method = method
        self.config = config or {}
        self.device = self._get_device(device)
        
        # Initialize the specific stabilizer
        self.stabilizer = self._create_stabilizer()
        
        logger.info(f"Initialized VideoStabilizer with method: {method}")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best available device."""
        if device:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _create_stabilizer(self):
        """Create the appropriate stabilizer based on method."""
        if self.method == "optical_flow":
            return OpticalFlowStabilizer(self.config, self.device)
        elif self.method == "homography":
            return HomographyStabilizer(self.config, self.device)
        elif self.method == "deep":
            return DeepStabilizer(self.config, self.device)
        else:
            raise ValueError(f"Unknown stabilization method: {self.method}")
    
    def stabilize(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Stabilize a video file.
        
        Args:
            input_path: Path to input video file
            output_path: Optional path to save stabilized video
            **kwargs: Additional arguments passed to the stabilizer
            
        Returns:
            Stabilized video frames as numpy array
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        logger.info(f"Stabilizing video: {input_path}")
        
        # Load video
        processor = VideoProcessor()
        frames = processor.load_video(input_path)
        
        # Stabilize frames
        stabilized_frames = self.stabilizer.stabilize(frames, **kwargs)
        
        # Save if output path provided
        if output_path:
            self.save_video(stabilized_frames, output_path, processor.fps)
        
        logger.info("Video stabilization completed")
        return stabilized_frames
    
    def save_video(
        self,
        frames: np.ndarray,
        output_path: Union[str, Path],
        fps: float = 30.0,
    ) -> None:
        """Save stabilized frames as video file.
        
        Args:
            frames: Video frames as numpy array
            output_path: Path to save the video
            fps: Frames per second for output video
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        height, width = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Saved stabilized video to: {output_path}")
    
    def get_metrics(
        self,
        original_frames: np.ndarray,
        stabilized_frames: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate stabilization quality metrics.
        
        Args:
            original_frames: Original video frames
            stabilized_frames: Stabilized video frames
            
        Returns:
            Dictionary of metric names and values
        """
        from .utils import MetricsCalculator
        
        calculator = MetricsCalculator()
        return calculator.calculate_all_metrics(original_frames, stabilized_frames)
