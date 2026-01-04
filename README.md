# 3D Scene Understanding with SpatialLM

A Python application that processes 3D point cloud data from SLAM systems and enables natural language querying of spatial environments using the SpatialLM model.

## Overview

This project bridges the gap between 3D reconstruction and natural language understanding. It automatically segments point clouds into semantic components (floors, walls, objects), then leverages SpatialLM to answer questions about the spatial layout in plain English.

## Features

- **Automatic Scene Segmentation**: Uses RANSAC plane detection to identify floors and walls
- **Object Clustering**: DBSCAN-based clustering to locate distinct objects in the scene
- **Natural Language Interface**: Ask questions about your 3D environment in plain English
- **GPU Acceleration**: Optional GPU support for faster inference
- **Plug-and-Play**: Works with standard PLY point cloud files from SLAM systems

## Requirements
```bash
pip install open3d numpy scikit-learn transformers torch
```

For GPU support, ensure you have CUDA-enabled PyTorch installed.

## Project Structure
```
.
├── app/
│   ├── scene_processing.py    # Point cloud segmentation and clustering
│   ├── spatiallm_wrapper.py   # SpatialLM model wrapper
│   └── main.py                # Interactive CLI application
└── output/
    └── room_pointcloud.ply    # Your SLAM-generated point cloud
```

## Usage

### Basic Example
```python
from scene_processing import build_scene_description_from_pcd
from spatiallm_wrapper import SpatialLMClient

# Load and process your point cloud
scene_desc = build_scene_description_from_pcd("output/room_pointcloud.ply")

# Initialize the spatial reasoning model
client = SpatialLMClient(use_gpu=True)

# Ask questions about your space
answer = client.ask(scene_desc, "Where is the desk located?")
print(answer)
```

### Interactive Mode

Run the main application for an interactive Q&A session:
```bash
python app/main.py
```

## How It Works

### 1. Scene Processing (`scene_processing.py`)

- **`segment_floor_and_walls()`**: Applies RANSAC plane fitting to extract the dominant floor plane and up to two vertical wall planes
- **`cluster_objects()`**: Uses DBSCAN to group remaining points into distinct objects and computes their centroids
- **`build_scene_description_from_pcd()`**: Orchestrates the pipeline and generates a text description of the scene

### 2. SpatialLM Interface (`spatiallm_wrapper.py`)

- Wraps the SpatialLM1.1-Llama-1B model from Hugging Face
- Handles model initialization with optional 8-bit quantization for GPU efficiency
- Provides a simple `ask()` method for spatial reasoning queries

### 3. Main Application (`main.py`)

- Loads point cloud data from your SLAM system
- Initializes the scene understanding pipeline
- Provides an interactive command-line interface for queries


## Model Information

This project uses [SpatialLM1.1-Llama-1B](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B), a language model fine-tuned for spatial reasoning tasks. The model can:
- Understand spatial relationships between objects
- Answer questions about object locations
- Reason about scene geometry and layout



## Future Enhancements

- [ ] Semantic object labeling (chair, desk, etc.)
- [ ] Support for more complex wall configurations
- [ ] Integration with real-time SLAM systems
- [ ] Web-based visualization interface
- [ ] Multi-room scene understanding

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.



## Acknowledgments

- Built with [Open3D](http://www.open3d.org/) for 3D processing
- Powered by [SpatialLM](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B) for spatial reasoning
- Uses scikit-learn for clustering algorithms

