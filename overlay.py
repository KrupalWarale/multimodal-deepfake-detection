import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Any, Dict, List, Tuple

# Initialize MediaPipe FaceMesh components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- COLOR DEFINITIONS (BGR format) ---
BASE_COLOR = (0, 255, 0)  # Parrot Green for all landmarks and connections

FEATURE_COLORS = {
    mp_face_mesh.FACEMESH_LIPS: BASE_COLOR,
    mp_face_mesh.FACEMESH_LEFT_EYE: BASE_COLOR,
    mp_face_mesh.FACEMESH_RIGHT_EYE: BASE_COLOR,
    mp_face_mesh.FACEMESH_LEFT_IRIS: BASE_COLOR,
    mp_face_mesh.FACEMESH_RIGHT_IRIS: BASE_COLOR,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW: BASE_COLOR,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW: BASE_COLOR
}

LINE_COLOR = BASE_COLOR  # Same parrot green for all connection lines

# Combine all connection sets for convenience
ALL_CONNECTIONS = tuple(
    list(mp_face_mesh.FACEMESH_LIPS) +
    list(mp_face_mesh.FACEMESH_LEFT_EYE) +
    list(mp_face_mesh.FACEMESH_RIGHT_EYE) +
    list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) +
    list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW) +
    list(mp_face_mesh.FACEMESH_IRISES)
)


class FaceOverlayProcessor:
    """Process video frames and overlay a uniform Parrot Green face mesh."""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _draw_custom_overlay(self, frame: np.ndarray, face_landmarks: Any, confidence: float):
        """Draws all landmarks and connecting lines in Parrot Green."""
        h, w = frame.shape[:2]

        # Draw dots for all landmarks
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, BASE_COLOR, -1)  # radius=1

        # Draw connecting lines for all features
        for feature in FEATURE_COLORS.keys():
            line_spec = mp_drawing.DrawingSpec(color=LINE_COLOR, thickness=1)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=feature,
                landmark_drawing_spec=None,
                connection_drawing_spec=line_spec
            )

    def process_frame_with_overlay(self, frame: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        """Processes a single frame and adds the green face mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        overlay_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._draw_custom_overlay(overlay_frame, face_landmarks, confidence)

        return overlay_frame

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
        cv2.putText(overlay_frame, "PARROT GREEN OVERLAY ACTIVE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Overlay', overlay_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
