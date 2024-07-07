# Multiple Real-Time Camera Tracking System

## 1. Description

This system enables real-time object tracking across multiple video sources using two cameras. It integrates object detection, tracking, and person re-identification (ReID) techniques to maintain consistent IDs for individuals moving between cameras or temporarily obscured.

## 2. Pipeline

![alt text](pipeline.png)

The system follows a modular pipeline:

### 2.1 Module Detection

- **Objective:** Detect objects (primarily people) using a pre-trained YOLOv8 model (`yolov8n.pt` by default) on each frame from all connected video sources.
- **Output:** Provides bounding boxes and corresponding confidence scores for detected objects.

### 2.2 Module Tracking

- **Objective:** Track detected objects within each video source using a bytetrack tracker (`bytetrack.yaml` by default).
- **Functionality:** Associates detections across frames to maintain tracklets (temporary object trajectories) and assigns unique IDs within each camera view.

### 2.3 Module Person ReID

- **Objective:** Re-identify individuals across different cameras or when they are temporarily obscured.
- **Implementation:** Uses an OSNet-based ReID model (`osnet_x0_75` by default) to extract features from detected object crops.
- **Matching:** Compares extracted features with stored features of known IDs using a distance-based approach (threshold of 600 by default).
- **Integration:** Merges tracklets when ReID confirms object identity across cameras, ensuring consistent ID assignment throughout the system.

## 3. Implementation

### Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy (np)
- Ultralytics (for YOLOv8)

### Steps

1. **Adjust Configuration:**
   - Modify `det_model`, `tracker`, `reid_model`, `source1`, `source2`, and `threshold` arguments in the `main` function (`main.py`) 

2. **Configuration Parameters:**
   - `det_model`: Path to your YOLOv8 detection model.
   - `tracker`: Path to your tracking configuration file (e.g., `bytetrack.yaml`).
   - `reid_model`: Path to your ReID model weights.
   - `source1`, `source2`: Video source paths (webcam index, file paths, or camera stream URLs).
   - `threshold`: Distance threshold for ReID-based ID merging.

3. **Run the Script:**
   - Execute `python main.py` to start the real-time tracking system.
