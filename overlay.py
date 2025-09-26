import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Any, Dict, List, Tuple

# Initialize MediaPipe FaceMesh components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- COLOR DEFINITIONS (BGR format) ---
BASE_COLOR = (180, 50, 20)         # Deep Indigo (default dots)
LINE_COLOR = (200, 200, 200)       # Light Gray (subtle connections)

# Feature-specific colors
FEATURE_COLORS = {
    mp_face_mesh.FACEMESH_LIPS: (255, 99, 71),        # Tomato Red
    mp_face_mesh.FACEMESH_LEFT_EYE: (50, 205, 50),    # Lime Green
    mp_face_mesh.FACEMESH_RIGHT_EYE: (30, 144, 255),  # Dodger Blue
    mp_face_mesh.FACEMESH_LEFT_IRIS: (255, 255, 255), # White
    mp_face_mesh.FACEMESH_RIGHT_IRIS: (255, 255, 255),# White
    mp_face_mesh.FACEMESH_LEFT_EYEBROW: (255, 215, 0),# Gold
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW: (255, 215, 0)# Gold
}

# Combine all connection sets
ALL_CONNECTIONS = tuple(
    list(mp_face_mesh.FACEMESH_LIPS) +
    list(mp_face_mesh.FACEMESH_LEFT_EYE) +
    list(mp_face_mesh.FACEMESH_RIGHT_EYE) +
    list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) +
    list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW) +
    list(mp_face_mesh.FACEMESH_IRISES)
)


class FaceOverlayProcessor:
    """Process video frames and overlay a simplified, brightly colored face mesh."""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._landmark_to_color_map = self._map_landmarks_to_color()

    def _map_landmarks_to_color(self) -> Dict[int, Tuple[int, int, int]]:
        """Maps every landmark in a feature region to its designated color."""
        mapping = {}
        for region_connections, color in FEATURE_COLORS.items():
            indices = set([i for conn in region_connections for i in conn])
            for idx in indices:
                mapping[idx] = color
        return mapping

    def _draw_custom_overlay(self, frame: np.ndarray, face_landmarks: Any, confidence: float):
        """Draws colored dots + light connection lines."""
        h, w = frame.shape[:2]

        # 1. Draw dots for all landmarks
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            color = self._landmark_to_color_map.get(i, BASE_COLOR)
            cv2.circle(frame, (x, y), 1, color, -1)

        # 2. Draw subtle gray connections
        line_spec = mp_drawing.DrawingSpec(color=LINE_COLOR, thickness=1)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=ALL_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=line_spec
        )

    def process_frame_with_overlay(self, frame: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        """Processes a single frame and adds the simplified, colored face mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        overlay_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_custom_overlay(overlay_frame, face_landmarks, confidence)

        return overlay_frame

    def extract_face_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract face features for analysis."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            x_coords, y_coords = landmarks[:, 0], landmarks[:, 1]
            face_area = (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords))
            fake_prob = 0.5
            landmarks_with_confidence = [[float(lm[0]), float(lm[1]), float(lm[2]), fake_prob] for lm in landmarks]

            return {
                'landmarks': landmarks_with_confidence,
                'landmarks_sample': [landmarks[i].tolist() for i in [0, 10, 50] if i < len(landmarks)],
                'face_area': float(face_area),
                'landmarks_count': len(landmarks),
                'confidence': fake_prob
            }
        return {'landmarks': [], 'landmarks_sample': [], 'face_area': 0.0, 'landmarks_count': 0, 'confidence': 0.0}

    def process_video_with_overlays(self, video_path: str, output_dir: str,
                                    confidences: List[float] = None) -> List[Dict[str, Any]]:
        """Process entire video and generate frames with overlays."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count, face_features_list = 0, []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            confidence = confidences[frame_count] if confidences and frame_count < len(confidences) else 0.5
            overlay_frame = self.process_frame_with_overlay(frame, confidence)

            face_features = self.extract_face_features(frame)
            face_features.update({'frame': frame_count, 'confidence': confidence})
            face_features_list.append(face_features)

            cv2.imwrite(os.path.join(output_dir, f"overlay_frame_{frame_count:04d}.jpg"), overlay_frame)
            frame_count += 1

        cap.release()
        return face_features_list


def generate_overlay_frames(video_path: str, output_dir: str,
                            video_analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """Helper function for compatibility."""
    try:
        processor = FaceOverlayProcessor()
        confidences = None
        if video_analysis_result and 'probabilities' in video_analysis_result:
            fake_prob = video_analysis_result['probabilities'].get('Fake', 0.5)
            confidences = [fake_prob] * 50

        face_features = processor.process_video_with_overlays(video_path, output_dir, confidences)

        return {
            'success': True,
            'overlay_frames_dir': output_dir,
            'face_features_array': face_features,
            'count': len(face_features)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# --- Testing with Webcam ---
if __name__ == "__main__":
    processor = FaceOverlayProcessor()
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        overlay_frame = processor.process_frame_with_overlay(frame, 0.5)
        cv2.putText(overlay_frame, "FEATURE COLORS ACTIVE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Face Overlay', overlay_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
