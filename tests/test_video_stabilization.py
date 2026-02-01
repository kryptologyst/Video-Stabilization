"""Tests for video stabilization package."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from video_stabilization import VideoStabilizer
from video_stabilization.models import OpticalFlowStabilizer, HomographyStabilizer, DeepStabilizer
from video_stabilization.utils import (
    MetricsCalculator, 
    VideoProcessor, 
    create_synthetic_shaky_video,
    get_device,
    set_seed
)


class TestVideoProcessor:
    """Test VideoProcessor class."""
    
    def test_create_synthetic_video(self):
        """Test synthetic video creation."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            create_synthetic_shaky_video(tmp_path, num_frames=10, shake_intensity=5.0)
            assert Path(tmp_path).exists()
            
            processor = VideoProcessor()
            frames = processor.load_video(tmp_path)
            assert len(frames) == 10
            assert frames.shape[1:] == (480, 640, 3)  # height, width, channels
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_video_loading_saving(self):
        """Test video loading and saving."""
        # Create synthetic video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            create_synthetic_shaky_video(tmp_path, num_frames=5, shake_intensity=3.0)
            
            # Load video
            processor = VideoProcessor()
            frames = processor.load_video(tmp_path)
            
            # Save video
            output_path = tmp_path.replace(".mp4", "_output.mp4")
            processor.save_video(frames, output_path)
            
            assert Path(output_path).exists()
            
            # Load saved video and compare
            saved_frames = processor.load_video(output_path)
            assert len(saved_frames) == len(frames)
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            Path(tmp_path.replace(".mp4", "_output.mp4")).unlink(missing_ok=True)


class TestStabilizers:
    """Test stabilization methods."""
    
    def test_optical_flow_stabilizer(self):
        """Test optical flow stabilizer."""
        config = {"pyr_scale": 0.5, "levels": 3}
        stabilizer = OpticalFlowStabilizer(config, "cpu")
        
        # Create test frames
        frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        
        stabilized = stabilizer.stabilize(frames)
        assert stabilized.shape == frames.shape
        assert len(stabilized) == len(frames)
    
    def test_homography_stabilizer(self):
        """Test homography stabilizer."""
        config = {"nfeatures": 100, "ransac_threshold": 5.0}
        stabilizer = HomographyStabilizer(config, "cpu")
        
        # Create test frames
        frames = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        
        stabilized = stabilizer.stabilize(frames)
        assert stabilized.shape == frames.shape
        assert len(stabilized) == len(frames)
    
    def test_deep_stabilizer(self):
        """Test deep learning stabilizer."""
        config = {"input_channels": 6, "hidden_dim": 64}
        stabilizer = DeepStabilizer(config, "cpu")
        
        # Create test frames
        frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        
        stabilized = stabilizer.stabilize(frames)
        assert stabilized.shape == frames.shape
        assert len(stabilized) == len(frames)


class TestMetricsCalculator:
    """Test metrics calculation."""
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        calculator = MetricsCalculator()
        
        # Create test frames
        original = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        stabilized = original + np.random.randint(-10, 10, original.shape, dtype=np.int8)
        stabilized = np.clip(stabilized, 0, 255).astype(np.uint8)
        
        metrics = calculator.calculate_all_metrics(original, stabilized)
        
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "temporal_consistency_original" in metrics
        assert "temporal_consistency_stabilized" in metrics
        assert "temporal_improvement" in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics["ssim"] <= 1
        assert metrics["psnr"] > 0


class TestVideoStabilizer:
    """Test main VideoStabilizer class."""
    
    def test_initialization(self):
        """Test stabilizer initialization."""
        stabilizer = VideoStabilizer(method="optical_flow")
        assert stabilizer.method == "optical_flow"
        assert stabilizer.stabilizer is not None
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_stabilization(self):
        """Test end-to-end stabilization pipeline."""
        # Create synthetic video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            create_synthetic_shaky_video(tmp_path, num_frames=10, shake_intensity=8.0)
            
            # Initialize stabilizer
            stabilizer = VideoStabilizer(method="optical_flow")
            
            # Stabilize video
            stabilized_frames = stabilizer.stabilize(tmp_path)
            
            # Calculate metrics
            processor = VideoProcessor()
            original_frames = processor.load_video(tmp_path)
            
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.calculate_all_metrics(original_frames, stabilized_frames)
            
            # Basic assertions
            assert len(stabilized_frames) == len(original_frames)
            assert stabilized_frames.shape == original_frames.shape
            assert "psnr" in metrics
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
