import cv2
import mediapipe as mp
import numpy as np
import os
from typing import List, Tuple, Dict, Any
import json

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face mesh connections for different regions
# These are approximate landmark indices for different facial regions
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 279, 360, 460, 305, 392, 369, 400, 379, 365, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

class FaceOverlayProcessor:
    """Process video frames and overlay face mesh with deepfake confidence visualization"""
    
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_frame_with_overlay(self, frame: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        """
        Process a single frame and add face mesh overlay with confidence-based coloring
        
        Args:
            frame: Input frame as numpy array
            confidence: Deepfake confidence score (0.0 to 1.0)
            
        Returns:
            Frame with face overlay
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        overlay_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks as colored dots with region-based coloring
                self._draw_region_colored_face_landmarks_dots(overlay_frame, face_landmarks, confidence)
        
        return overlay_frame
    
    def _get_region_color(self, index: int, confidence: float) -> Tuple[int, int, int]:
        """
        Get color for a specific landmark based on its region and confidence
        
        Args:
            index: Landmark index (0-467)
            confidence: Deepfake confidence score (0.0 to 1.0)
            
        Returns:
            BGR color tuple
        """
        # Determine region based on landmark index
        if index in EYES:
            # Blue for eyes
            base_color = (255, 0, 0)  # Blue in BGR
        elif index in NOSE:
            # Green for nose
            base_color = (0, 255, 0)  # Green in BGR
        elif index in MOUTH:
            # Red for mouth
            base_color = (0, 0, 255)  # Red in BGR
        else:
            # Yellow for face outline and other features
            base_color = (0, 255, 255)  # Yellow in BGR
        
        # Adjust color intensity based on confidence
        # For fake (high confidence): more intense colors
        # For real (low confidence): less intense colors
        intensity_factor = 0.7 + 0.3 * confidence  # Range from 0.7 to 1.0
        
        adjusted_color = (
            int(base_color[0] * intensity_factor),
            int(base_color[1] * intensity_factor),
            int(base_color[2] * intensity_factor)
        )
        
        return adjusted_color
    
    def _draw_region_colored_face_landmarks_dots(self, frame: np.ndarray, face_landmarks: Any, confidence: float):
        """Draw face landmarks as colored dots with region-based coloring"""
        h, w = frame.shape[:2]
        
        # Draw all 468 landmarks as dots with region-based colors
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # Get region-specific color for each landmark
            color = self._get_region_color(i, confidence)
            # Draw a small circle (dot) for each landmark
            cv2.circle(frame, (x, y), 2, color, -1)
    
    def extract_face_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract face features for analysis
        
        Returns:
            Dictionary with face features
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            
            # Calculate face area (simplified)
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            face_area = (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords))
            
            # Sample some landmarks for display
            sample_indices = [0, 10, 50, 100, 200, 300, 400, 450]
            landmarks_sample = [landmarks[i].tolist() for i in sample_indices if i < len(landmarks)]
            
            # Add confidence value to features
            fake_prob = 0.5  # Default confidence
            landmarks_with_confidence = []
            for i, landmark in enumerate(landmarks):
                landmarks_with_confidence.append([
                    float(landmark[0]), 
                    float(landmark[1]), 
                    float(landmark[2]),
                    fake_prob  # Add confidence as 4th value
                ])
            
            return {
                'landmarks': landmarks_with_confidence,
                'landmarks_sample': landmarks_sample,
                'face_area': float(face_area),
                'landmarks_count': len(landmarks),
                'confidence': fake_prob
            }
        
        return {
            'landmarks': [],
            'landmarks_sample': [],
            'face_area': 0.0,
            'landmarks_count': 0,
            'confidence': 0.0
        }
    
    def process_video_with_overlays(self, video_path: str, output_dir: str, 
                                  confidences: List[float] = None) -> List[Dict[str, Any]]:
        """
        Process entire video and generate frames with overlays
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save overlay frames
            confidences: List of confidence scores for each frame
            
        Returns:
            List of face features for each frame
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        face_features_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get confidence for this frame (default to 0.5 if not provided)
            confidence = confidences[frame_count] if confidences and frame_count < len(confidences) else 0.5
            
            # Process frame with overlay
            overlay_frame = self.process_frame_with_overlay(frame, confidence)
            
            # Extract face features
            face_features = self.extract_face_features(frame)
            face_features['frame'] = frame_count
            # Update confidence in features
            face_features['confidence'] = confidence
            face_features_list.append(face_features)
            
            # Save frame
            output_path = os.path.join(output_dir, f"overlay_frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, overlay_frame)
            
            frame_count += 1
        
        cap.release()
        return face_features_list

def generate_overlay_frames(video_path: str, output_dir: str, 
                          video_analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate overlay frames for a video with deepfake analysis results
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save overlay frames
        video_analysis_result: Deepfake analysis results
        
    Returns:
        Dictionary with processing results
    """
    try:
        processor = FaceOverlayProcessor()
        
        # Extract confidences from video analysis if available
        confidences = None
        if video_analysis_result and 'probabilities' in video_analysis_result:
            # Use fake probability as confidence
            fake_prob = video_analysis_result['probabilities'].get('Fake', 0.5)
            # For demo purposes, we'll use the same confidence for all frames
            # In a real implementation, this would be frame-by-frame analysis
            confidences = [fake_prob] * 50  # Assuming 50 frames max
        
        # Process video with overlays
        face_features = processor.process_video_with_overlays(
            video_path, output_dir, confidences
        )
        
        return {
            'success': True,
            'overlay_frames_dir': output_dir,
            'face_features_array': face_features,
            'count': len(face_features)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# For testing purposes - this would typically be called from the Flask app
if __name__ == "__main__":
    # Example usage
    processor = FaceOverlayProcessor()
    
    # For webcam testing (uncomment to use)
    """
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with overlay (using 0.7 as example confidence)
        overlay_frame = processor.process_frame_with_overlay(frame, 0.7)
        
        cv2.imshow('Face Overlay', overlay_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    """
    
    print("FaceOverlayProcessor initialized. Ready to process frames with deepfake confidence overlays.")