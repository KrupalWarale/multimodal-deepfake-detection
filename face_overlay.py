#!/usr/bin/env python3
"""
Facial Overlay System
====================

A computer vision module for detecting faces and generating geometric overlays.
This system creates two types of visual overlays on detected faces:

1. Wireframe Overlays: Yellow geometric lines outlining facial features
   - Eyebrows, eyes, nose, mouth, and cheek contours
   - Clean geometric representation suitable for academic/research use

2. Landmark Dot Overlays: Blue dots marking specific facial landmarks
   - Precise point markers on key facial features
   - Includes highlighted pupils in yellow
   - Scalable dot sizes based on face dimensions

Features:
- Enhanced face detection with image preprocessing
- Adaptive feature positioning based on actual detected features
- Fallback to estimated positions when specific features aren't detected
- Duplicate face detection filtering
- Multi-scale detection for improved accuracy

Author: AI Assistant
Version: 1.0
Dependencies: OpenCV, NumPy
"""

# Import required libraries
import cv2        # Computer vision and image processing
import numpy as np # Numerical operations and array handling
import math       # Mathematical operations for geometric calculations

class FaceOverlay:
    """
    Face Detection and Overlay Generation System
    
    This class provides comprehensive face detection and overlay generation
    capabilities using OpenCV's Haar cascade classifiers. It can detect
    faces and specific facial features, then generate clean geometric
    overlays for visualization purposes.
    """
    
    def __init__(self):
        """
        Initialize the face detection system with multiple cascade classifiers
        
        Loads pre-trained Haar cascade classifiers for detecting:
        - Frontal faces (primary detection)
        - Eyes (for precise positioning)
        - Profile faces (additional coverage)
        - Mouths/smiles (for mouth positioning)
        """
        # Load Haar cascade classifiers from OpenCV's data directory
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
    
    def detect_faces(self, frame):
        """
        Detect faces in frame with enhanced accuracy using multiple detection passes
        
        This method uses advanced techniques to improve face detection:
        1. Image enhancement with CLAHE (Contrast Limited Adaptive Histogram Equalization)
        2. Multiple detection passes with different parameters
        3. Duplicate detection filtering to remove overlapping results
        
        Args:
            frame (numpy.ndarray): Input color image frame
            
        Returns:
            tuple: (enhanced_gray_image, detected_faces_array)
                - enhanced_gray_image: Preprocessed grayscale image
                - detected_faces_array: Array of face rectangles [(x, y, w, h), ...]
        """
        # Convert color image to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance image contrast for better detection accuracy
        # CLAHE improves detection in varying lighting conditions
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # First detection pass: Enhanced image with sensitive parameters
        faces1 = self.face_cascade.detectMultiScale(
            enhanced_gray, 
            scaleFactor=1.05,     # Small scale steps for better detection
            minNeighbors=6,       # Stricter neighbor requirement
            minSize=(30, 30),     # Minimum face size (30x30 pixels)
            maxSize=(500, 500),   # Maximum face size (500x500 pixels)
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Second detection pass: Original image with standard parameters
        faces2 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Standard scale factor
            minNeighbors=5,       # Standard neighbor requirement
            minSize=(40, 40),     # Slightly larger minimum size
            maxSize=(400, 400)    # Smaller maximum size
        )
        
        # Combine results from both detection passes
        all_faces = list(faces1) + list(faces2)
        if len(all_faces) == 0:
            return gray, []
        
        # Remove duplicate detections using overlap analysis
        filtered_faces = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            
            # Check against existing faces for overlap
            for existing in filtered_faces:
                ex, ey, ew, eh = existing
                
                # Calculate overlap area between current and existing face
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                face_area = w * h
                
                # If overlap is more than 50% of face area, consider it duplicate
                if overlap_area > face_area * 0.5:
                    is_duplicate = True
                    break
                    
            # Add face to results if it's not a duplicate
            if not is_duplicate:
                filtered_faces.append(face)
        
        return enhanced_gray, np.array(filtered_faces)
    
    def detect_facial_features(self, gray, face_rect):
        """
        Detect specific facial features within a detected face region
        
        This method performs targeted detection of eyes and mouth within
        the boundaries of a detected face, enabling more precise overlay
        positioning based on actual feature locations.
        
        Args:
            gray (numpy.ndarray): Grayscale image
            face_rect (tuple): Face bounding box (x, y, width, height)
            
        Returns:
            tuple: (detected_eyes, detected_mouths)
                - detected_eyes: List of eye rectangles in full image coordinates
                - detected_mouths: List of mouth rectangles in full image coordinates
        """
        x, y, w, h = face_rect
        
        # Extract face region of interest (ROI) for feature detection
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        # Size constraints are proportional to face size for better accuracy
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, 
            scaleFactor=1.1,                    # Standard scale factor
            minNeighbors=5,                     # Neighbor requirement
            minSize=(int(w*0.1), int(h*0.1)),   # Min: 10% of face dimensions
            maxSize=(int(w*0.3), int(h*0.2))    # Max: 30% width, 20% height
        )
        
        # Detect mouth/smile within the face region
        # Mouth is typically wider and in lower portion of face
        mouths = self.mouth_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,                    # Standard scale factor
            minNeighbors=5,                     # Neighbor requirement
            minSize=(int(w*0.2), int(h*0.1)),   # Min: 20% width, 10% height
            maxSize=(int(w*0.6), int(h*0.3))    # Max: 60% width, 30% height
        )
        
        # Convert ROI coordinates back to full image coordinates
        detected_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Add face offset to get absolute coordinates
            detected_eyes.append((x + ex, y + ey, ew, eh))
        
        detected_mouths = []
        for (mx, my, mw, mh) in mouths:
            # Add face offset to get absolute coordinates
            detected_mouths.append((x + mx, y + my, mw, mh))
        
        return detected_eyes, detected_mouths
    
    def draw_wireframe_overlay(self, frame):
        """
        Generate wireframe overlay with yellow geometric lines on detected faces
        
        Creates clean geometric overlays showing facial structure with:
        - Eyebrow lines positioned above detected or estimated eyes
        - Eye outlines around detected eye regions
        - Nose lines extending from eye area to mouth region
        - Cheek lines for facial contour definition
        - Mouth outline around detected or estimated mouth area
        
        Args:
            frame (numpy.ndarray): Input color image frame
            
        Returns:
            numpy.ndarray: Frame with yellow wireframe overlays drawn on detected faces
        """
        # Detect all faces in the frame
        gray, faces = self.detect_faces(frame)
        
        # Create a copy of the original frame for overlay drawing
        overlay_frame = frame.copy()
        
        # Process each detected face individually
        for (x, y, w, h) in faces:
            # Detect specific facial features within this face
            eyes, mouths = self.detect_facial_features(gray, (x, y, w, h))
            
            # Calculate face center and reference points
            center_x = x + w // 2  # Horizontal center of face
            center_y = y + h // 2  # Vertical center of face
            
            # Wireframe drawing parameters
            color = (0, 255, 255)  # BGR format: Yellow color
            thickness = 2          # Line thickness in pixels
            
            # Adaptive positioning: Use detected eyes if available, otherwise use estimates
            if len(eyes) >= 2:
                # Sort eyes by horizontal position (left to right in image)
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye = eyes_sorted[0]   # Left in image (person's right eye)
                right_eye = eyes_sorted[1]  # Right in image (person's left eye)
                
                # Draw eyebrows above detected eyes
                left_brow_x = left_eye[0] + left_eye[2] // 2
                left_brow_y = left_eye[1] - int(h * 0.08)
                right_brow_x = right_eye[0] + right_eye[2] // 2
                right_brow_y = right_eye[1] - int(h * 0.08)
                
                eyebrow_width = int(w * 0.12)
                
                # Left eyebrow
                cv2.line(overlay_frame, 
                        (left_brow_x - eyebrow_width//2, left_brow_y - 3),
                        (left_brow_x, left_brow_y), color, thickness)
                cv2.line(overlay_frame,
                        (left_brow_x, left_brow_y),
                        (left_brow_x + eyebrow_width//2, left_brow_y - 3), color, thickness)
                
                # Right eyebrow
                cv2.line(overlay_frame,
                        (right_brow_x - eyebrow_width//2, right_brow_y - 3),
                        (right_brow_x, right_brow_y), color, thickness)
                cv2.line(overlay_frame,
                        (right_brow_x, right_brow_y),
                        (right_brow_x + eyebrow_width//2, right_brow_y - 3), color, thickness)
                
                # Draw eye outlines around detected eyes
                for (ex, ey, ew, eh) in eyes:
                    eye_center_x = ex + ew // 2
                    eye_center_y = ey + eh // 2
                    cv2.ellipse(overlay_frame, (eye_center_x, eye_center_y), 
                              (int(ew * 0.6), int(eh * 0.8)), 0, 0, 360, color, thickness)
                
                # Calculate nose position based on eye positions
                nose_x = (left_brow_x + right_brow_x) // 2
                nose_top_y = max(left_eye[1] + left_eye[3], right_eye[1] + right_eye[3]) + int(h * 0.05)
                nose_bottom_y = nose_top_y + int(h * 0.15)
                
            else:
                # Fallback: Use proportional estimates when eyes aren't detected
                eyebrow_y = y + int(h * 0.25)
                eyebrow_width = int(w * 0.15)
                
                # Left eyebrow (viewer's right)
                left_brow_x = x + int(w * 0.25)
                cv2.line(overlay_frame, 
                        (left_brow_x - eyebrow_width//2, eyebrow_y - 5),
                        (left_brow_x + eyebrow_width//2, eyebrow_y), color, thickness)
                cv2.line(overlay_frame,
                        (left_brow_x + eyebrow_width//2, eyebrow_y),
                        (left_brow_x + eyebrow_width, eyebrow_y - 5), color, thickness)
                
                # Right eyebrow (viewer's left)
                right_brow_x = x + int(w * 0.75)
                cv2.line(overlay_frame,
                        (right_brow_x - eyebrow_width, eyebrow_y - 5),
                        (right_brow_x - eyebrow_width//2, eyebrow_y), color, thickness)
                cv2.line(overlay_frame,
                        (right_brow_x - eyebrow_width//2, eyebrow_y),
                        (right_brow_x + eyebrow_width//2, eyebrow_y - 5), color, thickness)
                
                # Draw eye outlines
                eye_y = y + int(h * 0.35)
                eye_width = int(w * 0.12)
                eye_height = int(h * 0.08)
                
                # Left eye (viewer's right)
                left_eye_x = x + int(w * 0.25)
                cv2.ellipse(overlay_frame, (left_eye_x, eye_y), (eye_width, eye_height), 0, 0, 360, color, thickness)
                
                # Right eye (viewer's left)  
                right_eye_x = x + int(w * 0.75)
                cv2.ellipse(overlay_frame, (right_eye_x, eye_y), (eye_width, eye_height), 0, 0, 360, color, thickness)
                
                nose_x = center_x
                nose_top_y = y + int(h * 0.45)
                nose_bottom_y = y + int(h * 0.65)
            
            # Draw nose lines
            nose_width = int(w * 0.08)
            cv2.line(overlay_frame,
                    (nose_x - nose_width, nose_top_y),
                    (nose_x - int(nose_width * 1.5), nose_bottom_y), color, thickness)
            cv2.line(overlay_frame,
                    (nose_x + nose_width, nose_top_y),
                    (nose_x + int(nose_width * 1.5), nose_bottom_y), color, thickness)
            
            # Draw cheek lines
            cheek_y = nose_bottom_y + int(h * 0.05)
            cheek_length = int(w * 0.18)
            
            # Left cheek line
            left_cheek_x = x + int(w * 0.15)
            cv2.line(overlay_frame,
                    (left_cheek_x, cheek_y - 8),
                    (left_cheek_x + cheek_length, cheek_y + 8), color, thickness)
            
            # Right cheek line
            right_cheek_x = x + int(w * 0.85)
            cv2.line(overlay_frame,
                    (right_cheek_x - cheek_length, cheek_y + 8),
                    (right_cheek_x, cheek_y - 8), color, thickness)
            
            # Use detected mouth if available, otherwise estimate
            if len(mouths) > 0:
                # Use the largest detected mouth
                mouth = max(mouths, key=lambda m: m[2] * m[3])
                mouth_x, mouth_y, mouth_w, mouth_h = mouth
                mouth_center_x = mouth_x + mouth_w // 2
                mouth_center_y = mouth_y + mouth_h // 2
                
                # Draw mouth outline around detected mouth
                cv2.ellipse(overlay_frame, (mouth_center_x, mouth_center_y), 
                          (int(mouth_w * 0.6), int(mouth_h * 0.8)), 0, 0, 360, color, thickness)
            else:
                # Fallback to estimated mouth position
                mouth_y = y + int(h * 0.75)
                mouth_width = int(w * 0.25)
                mouth_height = int(h * 0.08)
                cv2.ellipse(overlay_frame, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 360, color, thickness)
        
        return overlay_frame
    
    def draw_landmark_dots(self, frame):
        """
        Generate landmark dot overlay with blue dots marking facial features
        
        Creates precise point markers on facial landmarks including:
        - Forehead center point
        - Eyebrow points following natural arch shape
        - Eye contour points with highlighted pupils (yellow)
        - Nose bridge and tip points
        - Mouth contour points for upper and lower lips
        - Chin reference points
        
        Dot sizes are automatically scaled based on face dimensions for
        consistent appearance across different face sizes.
        
        Args:
            frame (numpy.ndarray): Input color image frame
            
        Returns:
            numpy.ndarray: Frame with blue landmark dots drawn on detected faces
        """
        # Detect all faces in the frame
        gray, faces = self.detect_faces(frame)
        
        # Create a copy of the original frame for dot drawing
        dot_frame = frame.copy()
        
        # Process each detected face individually
        for (x, y, w, h) in faces:
            # Detect specific facial features within this face
            eyes, mouths = self.detect_facial_features(gray, (x, y, w, h))
            
            # Calculate face center and reference points
            center_x = x + w // 2  # Horizontal center of face
            center_y = y + h // 2  # Vertical center of face
            
            # Landmark drawing parameters with adaptive sizing
            dot_color = (255, 0, 0)                    # BGR format: Blue color
            pupil_color = (0, 255, 255)                # BGR format: Yellow for pupils
            dot_radius = int(max(3, w * 0.012))        # Scale dot size with face width
            pupil_radius = int(max(2, w * 0.008))      # Smaller radius for pupils
            
            # Draw forehead reference point
            forehead_y = y + int(h * 0.12)  # 12% down from top of face
            cv2.circle(dot_frame, (center_x, forehead_y), dot_radius, dot_color, -1)
            
            # Adaptive positioning: Use detected eyes if available for precise placement
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]
                
                # Eyebrow dots based on detected eyes
                left_brow_x = left_eye[0] + left_eye[2] // 2
                left_brow_y = left_eye[1] - int(h * 0.06)
                right_brow_x = right_eye[0] + right_eye[2] // 2
                right_brow_y = right_eye[1] - int(h * 0.06)
                
                eyebrow_width = int(w * 0.08)
                
                # Left eyebrow dots
                left_eyebrow_positions = [
                    (left_brow_x - eyebrow_width, left_brow_y + 2),
                    (left_brow_x - eyebrow_width//2, left_brow_y - 1),
                    (left_brow_x, left_brow_y - 3),
                    (left_brow_x + eyebrow_width//2, left_brow_y - 1),
                    (left_brow_x + eyebrow_width, left_brow_y + 2),
                ]
                
                # Right eyebrow dots
                right_eyebrow_positions = [
                    (right_brow_x - eyebrow_width, right_brow_y + 2),
                    (right_brow_x - eyebrow_width//2, right_brow_y - 1),
                    (right_brow_x, right_brow_y - 3),
                    (right_brow_x + eyebrow_width//2, right_brow_y - 1),
                    (right_brow_x + eyebrow_width, right_brow_y + 2),
                ]
                
                for pos in left_eyebrow_positions + right_eyebrow_positions:
                    cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
                
                # Eye dots based on detected eyes
                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    eye_center_x = ex + ew // 2
                    eye_center_y = ey + eh // 2
                    
                    # Create eye landmark pattern
                    eye_positions = [
                        (eye_center_x - int(ew * 0.4), eye_center_y),  # Left corner
                        (eye_center_x - int(ew * 0.2), eye_center_y - int(eh * 0.3)),  # Top left
                        (eye_center_x, eye_center_y - int(eh * 0.4)),  # Top center (pupil)
                        (eye_center_x + int(ew * 0.2), eye_center_y - int(eh * 0.3)),  # Top right
                        (eye_center_x + int(ew * 0.4), eye_center_y),  # Right corner
                        (eye_center_x + int(ew * 0.2), eye_center_y + int(eh * 0.3)),  # Bottom right
                        (eye_center_x - int(ew * 0.2), eye_center_y + int(eh * 0.3)),  # Bottom left
                    ]
                    
                    for j, pos in enumerate(eye_positions):
                        if j == 2:  # Pupil position
                            cv2.circle(dot_frame, pos, pupil_radius, pupil_color, -1)
                        else:
                            cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
                
                # Calculate nose position based on eyes
                nose_x = (left_eye[0] + left_eye[2]//2 + right_eye[0] + right_eye[2]//2) // 2
                nose_top_y = max(left_eye[1] + left_eye[3], right_eye[1] + right_eye[3]) + int(h * 0.03)
                
            else:
                # Fallback: Use proportional estimates when eyes aren't detected
                eyebrow_y = y + int(h * 0.25)  # 25% down from top of face
                eyebrow_positions = [
                    (x + int(w * 0.2), eyebrow_y),
                    (x + int(w * 0.25), eyebrow_y - 3),
                    (x + int(w * 0.3), eyebrow_y),
                    (x + int(w * 0.35), eyebrow_y - 2),
                    (x + int(w * 0.4), eyebrow_y),
                    
                    (x + int(w * 0.6), eyebrow_y),
                    (x + int(w * 0.65), eyebrow_y - 2),
                    (x + int(w * 0.7), eyebrow_y),
                    (x + int(w * 0.75), eyebrow_y - 3),
                    (x + int(w * 0.8), eyebrow_y),
                ]
                
                for pos in eyebrow_positions:
                    cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
                
                # Eye dots
                eye_y = y + int(h * 0.35)
                eye_positions = [
                    # Left eye (viewer's right)
                    (x + int(w * 0.18), eye_y),
                    (x + int(w * 0.22), eye_y - 5),
                    (x + int(w * 0.25), eye_y - 7),  # Pupil
                    (x + int(w * 0.28), eye_y - 5),
                    (x + int(w * 0.32), eye_y),
                    (x + int(w * 0.28), eye_y + 5),
                    (x + int(w * 0.22), eye_y + 5),
                    
                    # Right eye (viewer's left)
                    (x + int(w * 0.68), eye_y + 5),
                    (x + int(w * 0.72), eye_y + 5),
                    (x + int(w * 0.75), eye_y - 7),  # Pupil
                    (x + int(w * 0.78), eye_y - 5),
                    (x + int(w * 0.82), eye_y),
                    (x + int(w * 0.78), eye_y + 5),
                    (x + int(w * 0.72), eye_y - 5),
                ]
                
                for i, pos in enumerate(eye_positions):
                    # Highlight pupils with yellow
                    if i == 2 or i == 9:  # Pupil positions
                        cv2.circle(dot_frame, pos, pupil_radius, pupil_color, -1)
                    else:
                        cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
                
                nose_x = center_x
                nose_top_y = y + int(h * 0.5)
            
            # Nose dots
            nose_positions = [
                (nose_x - int(w * 0.04), nose_top_y),
                (nose_x + int(w * 0.04), nose_top_y),
                (nose_x - int(w * 0.025), nose_top_y + int(h * 0.08)),
                (nose_x, nose_top_y + int(h * 0.1)),
                (nose_x + int(w * 0.025), nose_top_y + int(h * 0.08)),
            ]
            
            for pos in nose_positions:
                cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
            
            # Use detected mouth if available
            if len(mouths) > 0:
                mouth = max(mouths, key=lambda m: m[2] * m[3])
                mouth_x, mouth_y, mouth_w, mouth_h = mouth
                mouth_center_x = mouth_x + mouth_w // 2
                mouth_center_y = mouth_y + mouth_h // 2
                
                # Create mouth landmark pattern around detected mouth
                mouth_positions = [
                    # Upper lip
                    (mouth_center_x - int(mouth_w * 0.35), mouth_center_y - int(mouth_h * 0.2)),
                    (mouth_center_x - int(mouth_w * 0.2), mouth_center_y - int(mouth_h * 0.4)),
                    (mouth_center_x - int(mouth_w * 0.1), mouth_center_y - int(mouth_h * 0.3)),
                    (mouth_center_x, mouth_center_y - int(mouth_h * 0.2)),
                    (mouth_center_x + int(mouth_w * 0.1), mouth_center_y - int(mouth_h * 0.3)),
                    (mouth_center_x + int(mouth_w * 0.2), mouth_center_y - int(mouth_h * 0.4)),
                    (mouth_center_x + int(mouth_w * 0.35), mouth_center_y - int(mouth_h * 0.2)),
                    
                    # Lower lip
                    (mouth_center_x - int(mouth_w * 0.3), mouth_center_y + int(mouth_h * 0.2)),
                    (mouth_center_x - int(mouth_w * 0.15), mouth_center_y + int(mouth_h * 0.4)),
                    (mouth_center_x - int(mouth_w * 0.05), mouth_center_y + int(mouth_h * 0.45)),
                    (mouth_center_x, mouth_center_y + int(mouth_h * 0.5)),
                    (mouth_center_x + int(mouth_w * 0.05), mouth_center_y + int(mouth_h * 0.45)),
                    (mouth_center_x + int(mouth_w * 0.15), mouth_center_y + int(mouth_h * 0.4)),
                    (mouth_center_x + int(mouth_w * 0.3), mouth_center_y + int(mouth_h * 0.2)),
                ]
            else:
                # Fallback mouth position
                mouth_y = y + int(h * 0.75)
                mouth_positions = [
                    # Upper lip
                    (center_x - int(w * 0.12), mouth_y - 8),
                    (center_x - int(w * 0.08), mouth_y - 12),
                    (center_x - int(w * 0.04), mouth_y - 10),
                    (center_x, mouth_y - 8),
                    (center_x + int(w * 0.04), mouth_y - 10),
                    (center_x + int(w * 0.08), mouth_y - 12),
                    (center_x + int(w * 0.12), mouth_y - 8),
                    
                    # Lower lip
                    (center_x - int(w * 0.1), mouth_y + 8),
                    (center_x - int(w * 0.06), mouth_y + 12),
                    (center_x - int(w * 0.02), mouth_y + 14),
                    (center_x, mouth_y + 15),
                    (center_x + int(w * 0.02), mouth_y + 14),
                    (center_x + int(w * 0.06), mouth_y + 12),
                    (center_x + int(w * 0.1), mouth_y + 8),
                ]
            
            for pos in mouth_positions:
                cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
            
            # Chin dots
            chin_y = y + int(h * 0.9)
            chin_positions = [
                (center_x - int(w * 0.08), chin_y),
                (center_x, y + int(h * 0.95)),
                (center_x + int(w * 0.08), chin_y),
            ]
            
            for pos in chin_positions:
                cv2.circle(dot_frame, pos, dot_radius, dot_color, -1)
        
        return dot_frame
    
    def process_frame(self, frame):
        """
        Process a single frame to generate both types of facial overlays
        
        This is the main processing method that generates both wireframe
        and landmark dot overlays for a given frame. It's designed to be
        called from the main application for each frame that needs processing.
        
        Args:
            frame (numpy.ndarray): Input color image frame
            
        Returns:
            tuple: (wireframe_frame, landmark_frame)
                - wireframe_frame: Frame with yellow geometric wireframe overlays
                - landmark_frame: Frame with blue landmark dot overlays
        """
        # Generate wireframe overlay (yellow geometric lines)
        wireframe_frame = self.draw_wireframe_overlay(frame)
        
        # Generate landmark dot overlay (blue dots with yellow pupils)
        landmark_frame = self.draw_landmark_dots(frame)
        
        return wireframe_frame, landmark_frame