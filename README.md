# Interactive Virtual Assistant with Spatial Awareness

An AI-powered spatial assistant that understands 3D room layouts and answers natural language queries about object locations and spatial relationships.

---

##  Features

### Phase 1: MVP (Completed)
-  **Synthetic 3D Scene Generation**: Auto-generates a realistic room with furniture
-  **Spatial Reasoning Engine**: Answers queries like "Where is the chair?" with human-readable directions
-  **Interactive Visualization**: Real-time 3D highlighting with Open3D
-  **100% Offline**: No internet required, runs locally

### Phase 2: Real 3D Reconstruction (NEW! )
-  **Video-to-3D Pipeline**: Converts room videos to 3D point clouds
-  **Depth Estimation**: Using DepthAnything for accurate depth maps
-  **Automatic Object Detection**: YOLOv8 identifies furniture (chairs, sofas, tables, etc.)
-  **3D Object Mapping**: Projects 2D detections to 3D coordinates

---

##  Quick Start

### Prerequisites
- Python 3.12
- Virtual environment: `C:\venvs\3Dvenv` (or modify paths accordingly)

### Installation

```powershell
# Activate virtual environment
C:\venvs\3Dvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Synthetic Scene (Fast Demo)
```powershell
python spatial_assistant.py
```

#### Option 2: Real 3D from Video (Realistic)
```powershell
python spatial_assistant.py --video sample_room.mp4
```

---

##  How to Use

1. **Run the assistant** (choose synthetic or video mode)
2. **Wait for the 3D window** to appear showing the room
3. **Close the window** to begin interaction
4. **Ask questions**:
   - "Where is the chair?"
   - "How far is the sofa?"
   - "Show me free space."
5. **View highlights**: Objects you ask about will be highlighted with a **red bounding box + red sphere**


---

## 📁 Project Structure

```
3d-spatial-virtual-assistant/
│
├── spatial_assistant.py          # Main application (Synthetic + Video modes)
├── video_processor.py             # Video frame extraction
├── depth_estimator.py             # MiDaS depth estimation
├── object_detector_real.py        # YOLO object detection
├── real_scene_pipeline.py         # End-to-end video→3D pipeline
│
├── sample_room.mp4                # Example video for testing
├── requirements.txt               # Python dependencies
├── test_mvp.py                    # Unit tests
└── README.md                      # This file
```

---

## Testing

### Test Synthetic Scene
```powershell
python test_mvp.py
```

### Test Video Pipeline (Standalone)
```powershell
python real_scene_pipeline.py
```

---

##  Technical Details

### Architecture

#### Synthetic Mode
1. **Scene Generation**: Procedurally generate room with furniture
2. **Ground Truth Objects**: Known 3D positions for perfect demo
3. **Spatial Reasoning**: Geometric calculations for distance/direction
4. **Visualization**: Open3D rendering with highlights

#### Video Mode
1. **Video Processing**: Extract best frame from video
2. **Depth Estimation**: MiDaS converts RGB → depth map
3. **Point Cloud Creation**: Depth + RGB → 3D point cloud
4. **Object Detection**: YOLOv8 detects furniture in 2D
5. **3D Projection**: Map 2D detections to 3D using depth
6. **Spatial Reasoning**: Same engine as synthetic mode

### Key Components

- **Depth Anything**: State-of-the-art monocular depth estimation
- **YOLOv8**: Real-time object detection (nano model for speed)
- **Open3D**: 3D visualization and point cloud processing
- **Spatial Reasoning Engine**: Geometric calculations for natural language generation

---

## Configuration

### Performance Tuning

For **faster processing** (CPU-only):
- Uses `DepthAnything` (faster, slightly less accurate)
- Uses `yolov8n.pt` (nano model, fastest)
- Processing time: ~15-30 seconds per video

For **better quality** (GPU recommended):
- Edit `real_scene_pipeline.py` line 30: change `model_type="DPT_Large"`
- Edit `object_detector_real.py` line 81: change `model_name="yolov8x.pt"`
- Processing time: ~5-10 seconds with GPU

### Customization

**Add more object types**: Edit `object_detector_real.py` line 16-22 to include more COCO classes

**Adjust depth range**: Modify `max_depth` parameter in pipeline (default: 10 meters)

**Change user position**: Edit `spatial_assistant.py` line 160-161 to customize viewpoint

---

## Supported Object Types

Current YOLO detections:
- Chair 
- Sofa / Couch 
- Table (Dining Table) 
- Bed 
- TV

---




##  Future Enhancements

- [ ] Real-time SLAM with live camera feed
- [ ] Voice input/output integration
- [ ] Multi-room mapping
- [ ] Augmented reality overlay
- [ ] Mobile app deployment

---

**Made with ❤️**
