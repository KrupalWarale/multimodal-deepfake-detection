# Minimal Deepfake Detection System

## Installation and Setup

### Prerequisites
- Python 3.8+ with pip package manager
- Webcam (optional, for testing overlay functionality)

### Installation Steps
```bash
# 1. Clone or download the project
cd deepfake-detection-system

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Verify MediaPipe installation
python -c "import mediapipe as mp; print('MediaPipe installed successfully')"

# 4. Run the application
python minimal_app.py
```

### MediaPipe Installation Notes
- MediaPipe requires Python 3.8-3.11 (3.12+ not yet supported)
- On Windows, ensure Visual C++ redistributables are installed
- If installation fails, try: `pip install --upgrade pip` then retry
- For Apple Silicon Macs, use: `pip install mediapipe-silicon`

## Technical Architecture Overview

A lightweight Flask-based web application for individual modality deepfake detection using MobileNetV2+LSTM for video analysis, RawNetLite for audio analysis, and OpenCV-based face overlay visualization for enhanced frame-level analysis.

## Core Technical Components

### 1. Video Deepfake Detection Pipeline
```
Video Input → Frame Extraction → MobileNetV2 Feature Extraction → LSTM Temporal Analysis → Classification
```

### 2. Face Overlay Visualization Pipeline
```
Video Frames → Face Detection (MediaPipe Face Mesh) → Landmark Extraction → Confidence-Based Overlay → Real-time Display
```

**Face Overlay Features:**
- **Face Detection**: MediaPipe Face Mesh with 468 facial landmarks for precise detection
- **Landmark Visualization**: Real-time facial feature point extraction and display
- **Confidence Visualization**: Color-coded overlays based on deepfake confidence scores
  - Green: High confidence (likely real, >0.7)
  - Red: Low confidence (likely fake, <0.3)
  - Yellow: Uncertain (0.3-0.7)
- **Real-time Arrays**: Live display of facial landmark data for analysis
- **Frame Toggle**: Switch between original and overlay frames in web interface
- **Face Boundary**: Convex hull outline around detected faces with confidence coloring

**Implementation Details:**
- **Input Processing**: 15 consecutive frames at 224x224 resolution (optimized for speed)
- **Feature Extractor**: MobileNetV2 (ImageNet pretrained) with TimeDistributed wrapper
- **Temporal Modeling**: LSTM (128 hidden units) for sequential frame analysis
- **Output**: Binary classification (Real/Fake) with confidence scores
- **Frame-Level Analysis**: Individual confidence scores displayed on each frame

**Technical Flow:**
```python
# 1. Frame Preprocessing (Optimized)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (224, 224))  # Reduced from 299x299
frame_normalized = frame_resized.astype(np.float32) / 255.0

# 2. Model Architecture (Lightweight)
inputs = layers.Input(shape=(15, 224, 224, 3))  # Reduced from 30 frames
base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
base_model.trainable = False  # Frozen for speed
x = layers.TimeDistributed(base_model)(inputs)
x = layers.LSTM(128)(x)  # Reduced from 256
outputs = layers.Dense(2, activation='softmax')(x)

# 3. Face Overlay Processing
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
results = face_mesh.process(rgb_frame)
if results.multi_face_landmarks:
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    face_points = (landmarks[:, :2] * [w, h]).astype(int)
    # Color code based on deepfake confidence
    if confidence > 0.7:
        color = (0, 255, 0)  # Green for real
    elif confidence < 0.3:
        color = (0, 0, 255)  # Red for fake
    else:
        color = (0, 255, 255)  # Yellow for uncertain
    # Draw face mesh points
    for point in face_points[::10]:  # Every 10th point
        cv2.circle(overlay_frame, tuple(point), 3, color, -1)
    # Draw face boundary
    face_outline = cv2.convexHull(face_points)
    cv2.drawContours(overlay_frame, [face_outline], -1, color, 3)
```

### 2. Audio Deepfake Detection Pipeline
```
Audio Input → RawNetLite Processing → Convolutional Feature Extraction → GRU Temporal Analysis → Classification
```

**Implementation Details:**
- **Audio Preprocessing**: Resampling to 16kHz, 3-second clips (48000 samples)
- **Feature Extraction**: RawNetLite end-to-end neural network
- **Architecture**: 1D Conv + ResBlocks + Bidirectional GRU + FC layers
- **Output**: Binary classification with probability scores

**Technical Flow:**
```python
# 1. Audio Preprocessing
waveform, sr = torchaudio.load(audio_path)
processed_audio = preprocess_audio(waveform, sr, target_sr=16000, target_sec=3.0)

# 2. RawNetLite Architecture
class RawNetLite(nn.Module):
    def __init__(self):
        self.conv_pre = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64) 
        self.resblock3 = ResBlock(64)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.gru = nn.GRU(64, 128, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):  # x: [B, 1, 48000]
        x = self.conv_pre(x)  # [B, 64, 48000]
        x = self.resblocks(x) # [B, 64, 48000]
        x = self.pool(x)      # [B, 64, 64]
        x, _ = self.gru(x)    # [B, 64, 256]
        return torch.sigmoid(self.fc2(self.fc1(x[:, -1, :])))
```

### 3. Individual Analysis Display
```
Video Analysis → Individual Video Results
Audio Analysis → Individual Audio Results
```

**Analysis Strategy:**
- **No Fusion**: Video and audio results displayed separately
- **Frame-Level Confidence**: Overlay scores on each extracted frame
- **Modality-Specific**: Dedicated testing interfaces for each modality
- **Transparency**: Clear indication of model status (trained/untrained)

## System Architecture

### Flask Web Application Structure
```
minimal_app.py
├── Video Upload Handler (/upload)
├── Audio Upload Handler (/upload_audio)  # New
├── Video Processing Pipeline (process_video)
├── Face Overlay Processing (FaceOverlayProcessor)
├── Audio Testing Interface (/audio_test)  # New
├── Overlay Frame Serving (/overlay/<video_name>/<frame_name>)
├── Face Features API (/get_face_features/<video_name>)
├── Static File Serving (/frames, /audio)
├── Web Interface (/)
└── API Status Endpoint (/api/status)  # New
```

### Core Processing Workflow
```
1. File Upload → 2. Audio Extraction → 3. Frame Extraction → 4. Individual Analysis → 5. Face Overlay → 6. Result Display
```

**Detailed Process Flow:**
```python
def process_video(video_path):
    # 1. Audio Extraction (MoviePy)
    video_clip = mp.VideoFileClip(video_path)
    audio_path = extract_audio_as_mp3(video_clip)
    
    # 2. Frame Extraction (OpenCV) - Optimized
    cap = cv2.VideoCapture(video_path)
    frames = extract_frames_with_skip(cap, frame_skip=fps//2, max_frames=50)
    
    # 3. Individual Analysis (No Fusion)
    video_result = detect_video_only(video_path)
    audio_result = detect_audio_deepfake_rawnet(audio_path) if audio_path else None
    
    return {
        'video_analysis': video_result,
        'audio_analysis': audio_result  # Displayed separately
    }
```

## Web Interfaces

### 1. Main Interface (/)
- **Purpose**: Integrated video processing with frame extraction and face overlay
- **Features**: 
  - Video upload and processing
  - Frame-by-frame confidence display
  - Face overlay toggle (Original/Overlay frames)
  - Real-time face features array display
  - Audio playback with visual waveform
  - Dashboard-style results presentation
- **Analysis**: Both video and audio analysis (individual results) with face overlay visualization

### 2. Audio Test Interface (/audio_test)
- **Purpose**: Dedicated audio deepfake detection testing
- **Features**: Audio-only upload, RawNetLite analysis, system status checking
- **Analysis**: Audio-only deepfake detection with detailed results

## Technical Dependencies

### Core ML Libraries
- **TensorFlow 2.15.0+**: Deep learning framework for MobileNetV2
- **PyTorch 2.1.0+**: Backend for RawNetLite model
- **torchaudio 2.1.0+**: Audio processing for RawNetLite
- **MediaPipe 0.10.0+**: Face mesh detection and landmark extraction

### Video/Audio Processing
- **OpenCV 4.9.0.80**: Video frame extraction and manipulation
- **MoviePy 1.0.3**: Audio track extraction from video files
- **NumPy 1.26.4**: Numerical computations (version constrained for compatibility)

### Web Framework
- **Flask 3.0.0**: Web application framework and HTTP handling

## Model Specifications

### MobileNetV2+LSTM Configuration (Video)
```python
# Input Tensor Shape (Optimized)
input_shape = (batch_size, 15, 224, 224, 3)  # Reduced from 30 frames, 299x299

# Architecture Components
- TimeDistributed(MobileNetV2(weights='imagenet'))  # Lightweight feature extraction
- LSTM(128, return_sequences=False)                 # Reduced temporal modeling  
- Dropout(0.5)                                      # Regularization
- Dense(2, activation='softmax')                    # Classification layer

# Output Shape
output_shape = (batch_size, 2)  # [P(Real), P(Fake)]
```

### RawNetLite Configuration (Audio)
```python
# Input Tensor Shape
input_shape = (batch_size, 1, 48000)  # 3 seconds at 16kHz

# Architecture Components
class RawNetLite(nn.Module):
    - Conv1d(1, 64, kernel_size=3) + BatchNorm1d + ReLU     # Initial convolution
    - ResBlock(64) × 3                                      # Residual blocks
    - AdaptiveAvgPool1d(64)                                 # Temporal compression
    - GRU(64, 128, bidirectional=True)                     # Temporal modeling
    - Linear(256, 64) → Linear(64, 1) + Sigmoid            # Classification

# Output Shape
output_shape = (batch_size, 1)  # Probability of fake [0,1]
```

## Performance Characteristics

### Processing Metrics
- **Frame Extraction Rate**: ~2 frames per second (configurable, max 50 frames)
- **Video Processing**: Synchronous (blocking) with 5-minute timeout
- **Audio Processing**: ~2-5 seconds per 3-second audio clip
- **Model Loading Time**: ~5-10 seconds (first run, optimized)

### Scalability Constraints
- **Single-threaded**: Synchronous Flask application
- **Memory Optimized**: Reduced model sizes and frame counts
- **CPU Optimized**: Lightweight models for faster inference
- **Timeout Protection**: 5-minute processing limit with progress indicators

## File System Organization

### Runtime Directories
```
uploads/          # Uploaded video/audio files and extracted audio
static/frames/    # Extracted video frames (organized by video name)
templates/        # HTML templates for web interfaces
├── index.html        # Main video processing interface
└── audio_test.html   # Dedicated audio testing interface
models/           # Pre-trained model weights (optional)
```

### Processing Output Structure
```
video_name/
├── frame_0000.jpg  # Individual frames with confidence overlays
├── frame_0030.jpg
├── frame_0060.jpg
└── ...
```

## API Response Format

### Individual Analysis Results
```json
{
  "success": true,
  "video_analysis": {
    "predicted_class": "Real|Fake",
    "confidence": 0.85,
    "probabilities": {"Real": 0.15, "Fake": 0.85}
  },
  "audio_analysis": {
    "predicted_class": "Real|Fake", 
    "confidence": 0.73,
    "probabilities": {"Real": 0.27, "Fake": 0.73},
    "note": "Using untrained model - results are for demonstration only"
  },
  "frames_count": 15,
  "video_duration": 5.2,
  "audio_duration": 5.2
}
```

### Audio-Only Analysis Results
```json
{
  "success": true,
  "audio_analysis": {
    "predicted_class": "Real|Fake",
    "confidence": 0.73,
    "probabilities": {"Real": 0.27, "Fake": 0.73}
  },
  "audio_duration": 3.0,
  "audio_url": "/audio/timestamp_filename.mp3"
}
```

## Technical Limitations

### Current Implementation Constraints
- **Model Weights**: RawNetLite may use untrained weights if pre-trained model is incompatible
- **Frame Processing**: Limited to 50 frames maximum for performance
- **Video Duration**: Processing limited to 2 minutes for optimal performance
- **Audio Clips**: Fixed 3-second analysis windows for RawNetLite

### Performance Optimizations
- **Lightweight Models**: MobileNetV2 instead of Xception for speed
- **Reduced Input Size**: 224x224 frames instead of 299x299
- **Frame Limiting**: Maximum 50 frames processed per video
- **Frozen Base Models**: Feature extractors frozen for faster inference

## System Status and Monitoring

### API Status Endpoint (/api/status)
```json
{
  "success": true,
  "features": {
    "video_processing": true,
    "audio_extraction": true,
    "video_deepfake_detection": true,
    "audio_deepfake_detection": true
  },
  "model_info": {
    "video_model": "MobileNetV2 + LSTM",
    "audio_model": "RawNetLite",
    "rawnet_model_available": false
  }
}
```

## Development Notes

### Code Structure Principles
- **Modular Design**: Separate modules for video and audio detection
- **Individual Analysis**: No multimodal fusion, results displayed separately
- **Graceful Fallback**: Robust error handling with model compatibility checks
- **Performance Focused**: Optimized for speed and responsiveness

### Key Files Structure
```
├── minimal_app.py           # Main Flask application (407 lines)
├── deepfake_detection.py    # Video analysis module (124 lines)
├── audio_detection.py       # Audio analysis module (154 lines)
├── audio_preprocessor.py    # Audio preprocessing utilities (193 lines)
├── RawNetLite.py           # RawNetLite model architecture (118 lines)
├── overlay.py              # Face overlay and feature extraction (279 lines)
├── templates/
│   ├── index.html          # Main interface with overlay toggle (1767 lines)
│   └── audio_test.html     # Audio testing interface (683 lines)
└── requirements.txt        # Dependencies including MediaPipe (11 lines)
```

### Extension Points
- **Model Weights**: Support for loading compatible pre-trained RawNetLite models
- **Batch Processing**: Multi-file analysis capabilities
- **GPU Acceleration**: CUDA support for faster inference
- **Real-time Processing**: Streaming analysis for live audio/video