#!/usr/bin/env python3
"""
Video Processing Web Application
===============================

A Flask-based web application for processing video files with the following features:
- Video upload and validation
- Audio extraction from video files
- Frame extraction at regular intervals
- Face detection and overlay generation (wireframe and landmark dots)
- Web interface with navigation controls for browsing frames

Author: AI Assistant
Version: 1.0
Dependencies: Flask, OpenCV, MoviePy, Werkzeug
"""

# Import required libraries
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os                    # File system operations
import cv2                   # Computer vision and image processing
import moviepy.editor as mp  # Video processing and audio extraction
from werkzeug.utils import secure_filename  # Secure file handling
import glob                  # File pattern matching
import json                  # JSON data handling
from face_overlay import FaceOverlay  # Custom face detection and overlay module

# Initialize Flask application
app = Flask(__name__)

# Configuration settings
app.config['UPLOAD_FOLDER'] = 'uploads'                    # Directory for uploaded files
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024      # 100MB file size limit

# Create required directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)    # For uploaded video files
os.makedirs('static/frames', exist_ok=True)                # For extracted frame images
os.makedirs('static/overlays', exist_ok=True)              # For face overlay images

# Initialize the face detection and overlay system
face_overlay = FaceOverlay()

@app.route('/')
def index():
    """
    Serve the main web interface
    
    Returns:
        HTML template for the video processing interface
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handle video file upload and processing
    
    Accepts POST requests with video files, validates them, and processes them
    to extract audio, frames, and generate face overlays.
    
    Returns:
        JSON response with processing results or error messages
        
    HTTP Status Codes:
        200: Success - video processed successfully
        400: Bad Request - no file provided or invalid file
        500: Internal Server Error - processing failed
    """
    # Validate that a video file was provided in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file to the uploads directory
        file.save(filepath)
        
        # Process the video synchronously and return results
        result = process_video_sync(filepath)
        return jsonify(result)
    
    # Return error if file upload failed for any other reason
    return jsonify({'error': 'File upload failed'}), 500

def process_video_sync(video_path):
    """
    Process video file synchronously to extract audio, frames, and generate face overlays
    
    This function performs the following operations:
    1. Extracts video metadata (resolution, FPS, duration)
    2. Extracts audio track and saves as MP3
    3. Extracts frames at regular intervals
    4. Detects faces in frames and generates overlay images
    5. Saves all processed data to organized directories
    
    Args:
        video_path (str): Path to the uploaded video file
        
    Returns:
        dict: Processing results containing:
            - success (bool): Whether processing completed successfully
            - message (str): Success message or error description
            - video metadata (fps, resolution, duration, etc.)
            - file paths and URLs for generated content
            - frame counts and processing statistics
    """
    try:
        print(f"Processing video: {video_path}")
        
        # Step 1: Extract video metadata using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file with OpenCV")
        
        # Get essential video properties
        fps = cap.get(cv2.CAP_PROP_FPS)                    # Frames per second
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # Video width in pixels
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # Video height in pixels
        duration = frame_count_total / fps if fps > 0 else 0  # Duration in seconds
        
        # Release the video capture object
        cap.release()
        
        print(f"Video info: {width}x{height}, {fps} fps, {duration:.2f}s, {frame_count_total} frames")
        
        # Step 2: Extract audio track using MoviePy
        base_name = os.path.splitext(os.path.basename(video_path))[0]  # Get filename without extension
        audio_path = None
        
        try:
            print("Extracting audio...")
            # Load video file with MoviePy
            video_clip = mp.VideoFileClip(video_path)
            
            # Check if video has an audio track
            if video_clip.audio is not None:
                # Define output path for audio file
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}.mp3")
                
                # Extract and save audio as MP3 (suppress verbose output)
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                print(f"✓ Audio extracted to: {audio_path}")
            else:
                print("No audio track found in video")
                
            # Clean up video clip object
            video_clip.close()
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            # Continue processing even if audio extraction fails
        
        # Step 3: Extract frames and create face overlays
        print("Extracting frames and creating face overlays...")
        
        # Initialize video capture for frame extraction
        cap = cv2.VideoCapture(video_path)
        
        # Initialize counters and tracking variables
        frame_count = 0          # Total frames processed
        saved_frames = 0         # Successfully saved frames
        faces_detected = 0       # Frames with detected faces
        
        # Create organized directory structure for output files
        frames_dir = os.path.join('static', 'frames', base_name)      # Original frames
        overlays_dir = os.path.join('static', 'overlays', base_name)  # Face overlay images
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)
        
        # Calculate frame skip interval to avoid processing too many frames
        # Extract approximately 2 frames per second, minimum 1 frame every 30
        frame_skip = max(1, int(fps // 2)) if fps > 0 else 30
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()  # Read next frame
            if not ret:              # End of video reached
                break
            
            # Only process frames at specified intervals
            if frame_count % frame_skip == 0:
                # Save original frame as JPEG
                frame_filename = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
                success = cv2.imwrite(frame_filename, frame)
                
                if success:
                    saved_frames += 1
                    
                    # Step 4: Generate face overlays for frames with detected faces
                    try:
                        # Detect faces in the current frame
                        gray, faces = face_overlay.detect_faces(frame)
                        
                        # Only create overlays if faces are detected
                        if len(faces) > 0:
                            faces_detected += 1
                            
                            # Generate both wireframe and landmark overlays
                            wireframe_frame, landmark_frame = face_overlay.process_frame(frame)
                            
                            # Save overlay images with descriptive filenames
                            wireframe_filename = os.path.join(overlays_dir, f'wireframe_{frame_count:04d}.jpg')
                            landmark_filename = os.path.join(overlays_dir, f'landmarks_{frame_count:04d}.jpg')
                            
                            # Write overlay images to disk
                            cv2.imwrite(wireframe_filename, wireframe_frame)
                            cv2.imwrite(landmark_filename, landmark_frame)
                            
                    except Exception as e:
                        print(f"Overlay creation failed for frame {frame_count}: {e}")
                        # Continue processing other frames even if one fails
                else:
                    print(f"Warning: Failed to write frame {frame_count}")
            
            frame_count += 1  # Increment frame counter
        
        # Clean up video capture object
        cap.release()
        
        # Print processing summary
        print(f"✓ Extracted {saved_frames} frames (every {frame_skip} frames)")
        print(f"✓ Created overlays for {faces_detected} frames with faces")
        
        # Step 5: Generate URLs for web interface (when in Flask app context)
        audio_url = None
        frames_url = None
        
        try:
            # Generate URL for audio file if it exists
            if audio_path:
                audio_filename = os.path.basename(audio_path)
                audio_url = url_for('serve_audio', filename=audio_filename)
            
            # Generate URL for frames list endpoint
            frames_url = url_for('get_frames_list', video_name=base_name)
        except RuntimeError:
            # Not in Flask application context - URLs will be generated by frontend
            pass
        
        # Return comprehensive processing results
        return {
            'success': True,
            'message': 'Video processed successfully',
            'audio_path': audio_path,           # Local file path for audio
            'audio_url': audio_url,             # Web URL for audio playback
            'frames_count': saved_frames,       # Number of extracted frames
            'total_frames': frame_count,        # Total frames in video
            'frames_dir': frames_dir,           # Directory containing frames
            'frames_url': frames_url,           # API endpoint for frame list
            'video_name': base_name,            # Base name for file organization
            'video_duration': duration,         # Video duration in seconds
            'fps': fps,                         # Frames per second
            'resolution': f"{width}x{height}",  # Video resolution
            'frame_skip': frame_skip            # Frame extraction interval
        }
        
    except Exception as e:
        # Handle any unexpected errors during processing
        print(f"Error processing video: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/audio/<filename>')
def serve_audio(filename):
    """
    Serve extracted audio files for web playback
    
    Args:
        filename (str): Name of the audio file to serve
        
    Returns:
        File response with audio content and appropriate headers
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/frames/<video_name>/<frame_name>')
def serve_frame(video_name, frame_name):
    """
    Serve individual frame images
    
    Args:
        video_name (str): Base name of the processed video
        frame_name (str): Specific frame filename to serve
        
    Returns:
        File response with frame image content
    """
    frames_dir = os.path.join('static', 'frames', video_name)
    return send_from_directory(frames_dir, frame_name)

@app.route('/overlays/<video_name>/<overlay_name>')
def serve_overlay(video_name, overlay_name):
    """
    Serve face overlay images (wireframe or landmark dots)
    
    Args:
        video_name (str): Base name of the processed video
        overlay_name (str): Specific overlay filename to serve
        
    Returns:
        File response with overlay image content
    """
    overlays_dir = os.path.join('static', 'overlays', video_name)
    return send_from_directory(overlays_dir, overlay_name)

@app.route('/get_frames/<video_name>')
def get_frames_list(video_name):
    """
    API endpoint to get list of extracted frames for a processed video
    
    Args:
        video_name (str): Base name of the processed video
        
    Returns:
        JSON response containing:
        - success (bool): Whether frames were found
        - frames (list): URLs for each frame image
        - count (int): Total number of frames
        - error (str): Error message if frames not found
    """
    frames_dir = os.path.join('static', 'frames', video_name)
    
    if os.path.exists(frames_dir):
        # Get all JPEG files in the frames directory, sorted by name
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        # Generate URLs for each frame
        frame_urls = [url_for('serve_frame', video_name=video_name, frame_name=f) for f in frame_files]
        
        return jsonify({
            'success': True,
            'frames': frame_urls,
            'count': len(frame_files)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Frames directory not found'
        })

@app.route('/get_overlays/<video_name>')
def get_overlays_list(video_name):
    """
    API endpoint to get list of face overlay images for a processed video
    
    Args:
        video_name (str): Base name of the processed video
        
    Returns:
        JSON response containing:
        - success (bool): Whether overlays were found
        - wireframes (list): URLs for wireframe overlay images
        - landmarks (list): URLs for landmark dot overlay images
        - wireframe_count (int): Number of wireframe overlays
        - landmark_count (int): Number of landmark overlays
        - error (str): Error message if overlays not found
    """
    overlays_dir = os.path.join('static', 'overlays', video_name)
    
    if os.path.exists(overlays_dir):
        # Get wireframe overlay files (yellow geometric lines)
        wireframe_files = sorted([f for f in os.listdir(overlays_dir) 
                                if f.startswith('wireframe_') and f.endswith('.jpg')])
        
        # Get landmark overlay files (blue dots)
        landmark_files = sorted([f for f in os.listdir(overlays_dir) 
                               if f.startswith('landmarks_') and f.endswith('.jpg')])
        
        # Generate URLs for each overlay type
        wireframe_urls = [url_for('serve_overlay', video_name=video_name, overlay_name=f) 
                         for f in wireframe_files]
        landmark_urls = [url_for('serve_overlay', video_name=video_name, overlay_name=f) 
                        for f in landmark_files]
        
        return jsonify({
            'success': True,
            'wireframes': wireframe_urls,
            'landmarks': landmark_urls,
            'wireframe_count': len(wireframe_files),
            'landmark_count': len(landmark_files)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Overlays directory not found'
        })

@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring application status
    
    Returns:
        JSON response with application status and version information
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Video processing app is running',
        'opencv_version': cv2.__version__
    })

if __name__ == '__main__':
    """
    Application entry point
    
    Starts the Flask development server with the following configuration:
    - Debug mode enabled for development
    - Accessible on all network interfaces (0.0.0.0)
    - Running on port 5000
    """
    print("Starting Minimal Video Processing App...")
    print("✓ All dependencies loaded successfully")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)