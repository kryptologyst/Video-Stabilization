# Video Stabilization

Advanced Video Stabilization with Deep Learning and Traditional Methods

## Overview

This project implements state-of-the-art video stabilization techniques including:

- **Traditional Methods**: Optical flow-based stabilization, homography estimation, mesh warping
- **Deep Learning Approaches**: Neural stabilization networks, temporal consistency models
- **Advanced Techniques**: Gyro-aware stabilization, content-aware warping, smoothness priors

## Features

- Multiple stabilization algorithms (optical flow, homography, deep learning)
- Comprehensive evaluation metrics (PSNR, SSIM, temporal consistency)
- Interactive demo with Streamlit/Gradio
- Production-ready code with proper testing and documentation
- Support for various video formats and resolutions
- Deterministic seeding for reproducibility
- Device fallback (CUDA → MPS → CPU)

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- OpenCV 4.8+

### Install from source

```bash
git clone https://github.com/kryptologyst/Video-Stabilization.git
cd Video-Stabilization
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from video_stabilization import VideoStabilizer
import cv2

# Initialize stabilizer
stabilizer = VideoStabilizer(method="optical_flow")

# Stabilize video
stabilized_video = stabilizer.stabilize("input_video.mp4")
stabilizer.save_video(stabilized_video, "output_video.mp4")
```

### Advanced Usage

```python
from video_stabilization import VideoStabilizer
from video_stabilization.utils import MetricsCalculator

# Initialize with custom config
config = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15
}
stabilizer = VideoStabilizer(method="optical_flow", config=config)

# Stabilize and evaluate
stabilized_frames = stabilizer.stabilize("input_video.mp4")

# Calculate metrics
metrics_calc = MetricsCalculator()
metrics = metrics_calc.calculate_all_metrics(original_frames, stabilized_frames)
print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.4f}")
```

## Demo Applications

### Streamlit Demo

```bash
streamlit run demo/app.py
```

### Command Line Demo

```bash
python scripts/demo.py --method optical_flow --create-synthetic
```

## Training

Train deep learning models for video stabilization:

```bash
python scripts/train.py --config configs/deep.yaml
```

## Evaluation

Evaluate different stabilization methods:

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
```

## Dataset

The project supports various video datasets:
- Custom video collections
- Synthetic shaky videos (generated automatically)
- Real-world handheld footage

### Creating Synthetic Data

```python
from video_stabilization.utils import create_synthetic_shaky_video

create_synthetic_shaky_video(
    "synthetic_video.mp4",
    num_frames=100,
    shake_intensity=10.0
)
```

## Metrics

The package provides comprehensive evaluation metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Temporal Consistency**: Frame-to-frame stability measure
- **Stabilization Quality**: Overall stabilization effectiveness

## Architecture

```
src/
├── models/          # Stabilization models
│   ├── base.py      # Base stabilizer class
│   └── __init__.py
├── data/           # Data loading and preprocessing
├── utils/          # Utility functions
│   ├── utils.py    # Core utilities
│   └── __init__.py
├── train/          # Training scripts
├── eval/           # Evaluation metrics
├── core.py         # Main VideoStabilizer class
└── __init__.py

configs/            # Configuration files
├── optical_flow.yaml
├── homography.yaml
├── deep.yaml
└── evaluation.yaml

scripts/            # Command line scripts
├── train.py
├── evaluate.py
└── demo.py

demo/               # Demo applications
└── app.py          # Streamlit demo

tests/              # Unit tests
└── test_video_stabilization.py

notebooks/          # Jupyter notebooks
└── demo.ipynb
```

## Configuration

The package uses YAML configuration files for easy customization:

### Optical Flow Configuration (`configs/optical_flow.yaml`)

```yaml
method: "optical_flow"
pyr_scale: 0.5
levels: 3
winsize: 15
iterations: 3
poly_n: 5
poly_sigma: 1.2
flags: 0
```

### Homography Configuration (`configs/homography.yaml`)

```yaml
method: "homography"
nfeatures: 1000
ransac_threshold: 5.0
```

### Deep Learning Configuration (`configs/deep.yaml`)

```yaml
method: "deep"
input_channels: 6
hidden_dim: 256
checkpoint_path: null
learning_rate: 0.001
batch_size: 8
num_epochs: 100
weight_decay: 1e-4
```

## API Reference

### VideoStabilizer

Main class for video stabilization.

```python
class VideoStabilizer:
    def __init__(self, method: str, config: Dict[str, Any], device: str)
    def stabilize(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]]) -> np.ndarray
    def save_video(self, frames: np.ndarray, output_path: Union[str, Path], fps: float)
    def get_metrics(self, original_frames: np.ndarray, stabilized_frames: np.ndarray) -> Dict[str, float]
```

### Stabilization Methods

#### OpticalFlowStabilizer

Uses Farneback optical flow for motion estimation and translation-based stabilization.

#### HomographyStabilizer

Uses feature matching and homography estimation for robust stabilization handling rotation and scaling.

#### DeepStabilizer

Uses neural networks to predict stabilization transformations.

### Utility Functions

#### VideoProcessor

```python
class VideoProcessor:
    def load_video(self, video_path: Union[str, Path]) -> np.ndarray
    def save_video(self, frames: np.ndarray, output_path: Union[str, Path], fps: Optional[float])
```

#### MetricsCalculator

```python
class MetricsCalculator:
    def calculate_psnr(self, original: np.ndarray, stabilized: np.ndarray) -> float
    def calculate_ssim(self, original: np.ndarray, stabilized: np.ndarray) -> float
    def calculate_temporal_consistency(self, frames: np.ndarray) -> float
    def calculate_all_metrics(self, original: np.ndarray, stabilized: np.ndarray) -> Dict[str, float]
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Development

### Code Formatting

```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Performance

The package is optimized for performance:

- **Device Support**: Automatic CUDA/MPS/CPU fallback
- **Memory Efficient**: Gradient checkpointing and mixed precision support
- **Batch Processing**: Support for batch video processing
- **Streaming**: Optional streaming loaders for large datasets

## Limitations

- Deep learning models require pre-trained weights for optimal performance
- Real-time processing may require GPU acceleration
- Very large videos may require chunked processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## Citation

If you use this project in your research, please cite:

```bibtex
@software{video_stabilization,
  title={Advanced Video Stabilization with Deep Learning and Traditional Methods},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Video-Stabilization}
}
```
# Video-Stabilization
