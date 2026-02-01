"""Streamlit demo for video stabilization."""

import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import torch

from video_stabilization import VideoStabilizer
from video_stabilization.utils import MetricsCalculator, VideoProcessor, create_synthetic_shaky_video, set_seed

# Page config
st.set_page_config(
    page_title="Video Stabilization Demo",
    page_icon="🎬",
    layout="wide"
)

# Set random seed
set_seed(42)


def load_video_frames(video_path: str) -> tuple[np.ndarray, float]:
    """Load video frames and return frames and fps."""
    processor = VideoProcessor()
    frames = processor.load_video(video_path)
    return frames, processor.fps


def create_video_preview(frames: np.ndarray, fps: float) -> str:
    """Create a preview video from frames."""
    temp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    processor = VideoProcessor()
    processor.save_video(frames, temp_path.name, fps)
    return temp_path.name


def main():
    """Main Streamlit app."""
    st.title("🎬 Video Stabilization Demo")
    st.markdown("Upload a video or create a synthetic shaky video to test different stabilization methods.")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Stabilization Method",
        ["optical_flow", "homography", "deep"],
        help="Choose the stabilization algorithm"
    )
    
    # Method descriptions
    method_descriptions = {
        "optical_flow": "Uses Farneback optical flow to estimate motion between frames and applies translation-based stabilization.",
        "homography": "Uses feature matching and homography estimation for robust stabilization that handles rotation and scaling.",
        "deep": "Uses a neural network to predict stabilization transformations (requires pre-trained model)."
    }
    
    st.sidebar.markdown(f"**{method.title()}**: {method_descriptions[method]}")
    
    # Video input options
    st.sidebar.subheader("Video Input")
    input_option = st.sidebar.radio(
        "Choose input method",
        ["Upload video", "Create synthetic shaky video"]
    )
    
    input_frames = None
    input_fps = 30.0
    
    if input_option == "Upload video":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video file to stabilize"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                input_frames, input_fps = load_video_frames(tmp_path)
                st.success(f"Loaded video: {len(input_frames)} frames, {input_fps:.1f} FPS")
            except Exception as e:
                st.error(f"Error loading video: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    
    else:  # Create synthetic video
        st.sidebar.subheader("Synthetic Video Parameters")
        num_frames = st.sidebar.slider("Number of frames", 50, 200, 100)
        shake_intensity = st.sidebar.slider("Shake intensity", 5.0, 30.0, 15.0)
        
        if st.sidebar.button("Generate Synthetic Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                create_synthetic_shaky_video(
                    tmp_path,
                    num_frames=num_frames,
                    shake_intensity=shake_intensity
                )
                input_frames, input_fps = load_video_frames(tmp_path)
                st.success(f"Generated synthetic video: {len(input_frames)} frames")
            except Exception as e:
                st.error(f"Error generating video: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    
    # Main content area
    if input_frames is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Video")
            
            # Show first frame
            st.image(input_frames[0], caption="First frame", use_column_width=True)
            
            # Video preview
            if len(input_frames) > 1:
                preview_path = create_video_preview(input_frames, input_fps)
                st.video(preview_path)
                Path(preview_path).unlink(missing_ok=True)
        
        with col2:
            st.subheader("Stabilized Video")
            
            # Stabilize video
            if st.button("Stabilize Video", type="primary"):
                with st.spinner("Stabilizing video..."):
                    try:
                        # Initialize stabilizer
                        stabilizer = VideoStabilizer(method=method)
                        
                        # Stabilize frames
                        stabilized_frames = stabilizer.stabilizer.stabilize(input_frames)
                        
                        # Show first stabilized frame
                        st.image(stabilized_frames[0], caption="First stabilized frame", use_column_width=True)
                        
                        # Show stabilized video preview
                        if len(stabilized_frames) > 1:
                            stabilized_preview_path = create_video_preview(stabilized_frames, input_fps)
                            st.video(stabilized_preview_path)
                            Path(stabilized_preview_path).unlink(missing_ok=True)
                        
                        # Calculate and display metrics
                        st.subheader("Stabilization Metrics")
                        metrics_calc = MetricsCalculator()
                        metrics = metrics_calc.calculate_all_metrics(input_frames, stabilized_frames)
                        
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("PSNR", f"{metrics['psnr']:.2f}")
                            st.metric("SSIM", f"{metrics['ssim']:.4f}")
                        
                        with col_metric2:
                            st.metric("Temporal Consistency", f"{metrics['temporal_consistency_stabilized']:.4f}")
                            st.metric("Improvement", f"{metrics['temporal_improvement']:.4f}")
                        
                        # Download stabilized video
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            processor = VideoProcessor()
                            processor.save_video(stabilized_frames, tmp_file.name, input_fps)
                            
                            with open(tmp_file.name, "rb") as f:
                                st.download_button(
                                    label="Download Stabilized Video",
                                    data=f.read(),
                                    file_name=f"stabilized_{method}.mp4",
                                    mime="video/mp4"
                                )
                            
                            Path(tmp_file.name).unlink(missing_ok=True)
                    
                    except Exception as e:
                        st.error(f"Error during stabilization: {e}")
    
    else:
        st.info("Please upload a video or generate a synthetic video to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Video Stabilization Demo** - Advanced Computer Vision Project | "
        "Supports optical flow, homography, and deep learning methods"
    )


if __name__ == "__main__":
    main()
