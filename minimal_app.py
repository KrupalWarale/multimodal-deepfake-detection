#!/usr/bin/env python3
"""
Minimal Video Processing and Deepfake Detection App
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import moviepy.editor as mp
import threading
import time
from werkzeug.utils import secure_filename
from deepfake_detection import detect_video_only
from audio_detection import detect_audio_deepfake_rawnet, is_audio_detection_available
from overlay import generate_overlay_frames, FaceOverlayProcessor  # Added overlay import

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FRAMES_FOLDER'] = os.path.join('static', 'frames')
app.config['STATIC_OVERLAY_FOLDER'] = os.path.join('static', 'overlay_frames')  # Added overlay folder
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FRAMES_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_OVERLAY_FOLDER'], exist_ok=True)  # Create overlay folder

class TimeoutException(Exception):
    pass

def run_with_timeout(func, args, timeout_duration):
    """Run a function with timeout"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        raise TimeoutException("Processing timeout")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        selected_model = 'cross_domain_focal_rawnet_lite.pt'  # Changed default model
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = process_video(filepath, selected_model)
        return jsonify(result)
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

def process_video_internal(video_path, selected_model=None):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count_total / fps if fps > 0 else 0
    
    max_duration = 120
    if duration > max_duration:
        print(f"Video too long ({duration:.1f}s), processing first {max_duration}s only")
    
    frames_dir = os.path.join(app.config['STATIC_FRAMES_FOLDER'], base_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_count = 0
    saved_frames = 0
    frame_skip = max(1, int(fps // 2)) if fps > 0 else 30
    max_frames_to_save = 50
    max_total_frames = int(fps * max_duration) if duration > max_duration else frame_count_total
    
    while True and saved_frames < max_frames_to_save:
        ret, frame = cap.read()
        if not ret or frame_count >= max_total_frames:
            break
        
        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
            if cv2.imwrite(frame_filename, frame):
                saved_frames += 1
        
        frame_count += 1
    
    cap.release()
    
    audio_path = None
    audio_url = None
    try:
        print("Extracting audio...")
        video_clip = mp.VideoFileClip(video_path)
        if video_clip.audio is not None:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}.mp3")
            if duration > max_duration:
                audio_clip = video_clip.audio.subclip(0, max_duration)
                audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
                audio_clip.close()
            else:
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            audio_url = f'/audio/{os.path.basename(audio_path)}'
        video_clip.close()
    except Exception as e:
        print(f"Audio extraction failed: {e}")
    
    print("Starting video analysis...")
    video_result = detect_video_only(video_path)
    
    # Generate overlay frames with deepfake analysis
    print("Generating overlay frames...")
    overlay_dir = os.path.join(app.config['STATIC_OVERLAY_FOLDER'], base_name)
    overlay_result = generate_overlay_frames(
        video_path, 
        overlay_dir, 
        video_result.get('video_analysis', {})
    )
    
    audio_result = None
    if audio_path and is_audio_detection_available():
        try:
            print(f"Starting audio analysis with RawNetLite using model: {selected_model}...")
            models_dir = os.path.join('models')
            if selected_model:
                model_path = os.path.join(models_dir, selected_model)
                if not os.path.exists(model_path):
                    model_path = os.path.join(models_dir, 'cross_domain_focal_rawnet_lite.pt')
            else:
                model_path = os.path.join(models_dir, 'cross_domain_focal_rawnet_lite.pt')
            
            audio_result = detect_audio_deepfake_rawnet(audio_path, model_path)
            audio_result['model_used'] = selected_model if selected_model else 'cross_domain_focal_rawnet_lite.pt'
            print("Audio analysis completed")
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            audio_result = {'predicted_class': 'Error', 'error': str(e)}
    elif not is_audio_detection_available():
        print("Audio analysis skipped: PyTorch dependencies not available")
    
    deepfake_result = {
        'video_analysis': video_result['video_analysis'],
        'audio_analysis': audio_result
    }
    
    print("Analysis completed")
    
    return {
        'success': True,
        'message': 'Video processed successfully',
        'frames_count': saved_frames,
        'total_frames': frame_count,
        'video_name': base_name,
        'video_duration': min(duration, max_duration),
        'original_duration': duration,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'frame_skip': frame_skip,
        'audio_url': audio_url,
        'audio_path': audio_path,
        'deepfake_analysis': deepfake_result,
        'overlay_analysis': overlay_result,  # Added overlay analysis results
        'processing_limited': duration > max_duration,
        'dashboard_summary': {
            'video_processed': True,
            'audio_extracted': audio_path is not None,
            'frames_extracted': saved_frames,
            'video_analysis_completed': True,
            'audio_analysis_completed': audio_result is not None and 'error' not in audio_result,
            'individual_results_only': True
        }
    }

def process_video(video_path, selected_model=None):
    try:
        return run_with_timeout(process_video_internal, (video_path, selected_model), 240)
    except TimeoutException:
        return {'success': False, 'error': 'Processing timeout - video too complex or too long'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/status')
def api_status():
    try:
        status = {
            'success': True,
            'features': {
                'video_processing': True,
                'audio_extraction': True,
                'video_deepfake_detection': True,
                'audio_deepfake_detection': is_audio_detection_available(),
                'face_overlay_visualization': True  # Added overlay feature
            },
            'model_info': {
                'video_model': 'MobileNetV2 + LSTM',
                'audio_model': 'RawNetLite' if is_audio_detection_available() else 'Not Available',
                'overlay_model': 'MediaPipe FaceMesh'
            }
        }
        
        model_path = os.path.join('models', 'cross_domain_focal_rawnet_lite.pt')  # Updated model name
        status['model_info']['rawnet_model_available'] = os.path.exists(model_path)
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/frames/<video_name>/<frame_name>')
def serve_frame(video_name, frame_name):
    frames_dir = os.path.join(app.config['STATIC_FRAMES_FOLDER'], video_name)
    return send_from_directory(frames_dir, frame_name)

@app.route('/overlay_frames/<video_name>/<frame_name>')  # Added overlay frames route
def serve_overlay_frame(video_name, frame_name):
    overlay_dir = os.path.join(app.config['STATIC_OVERLAY_FOLDER'], video_name)
    return send_from_directory(overlay_dir, frame_name)

@app.route('/get_frames/<video_name>')
def get_frames_list(video_name):
    frames_dir = os.path.join(app.config['STATIC_FRAMES_FOLDER'], video_name)
    
    if os.path.exists(frames_dir):
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        frame_urls = [f'/frames/{video_name}/{f}' for f in frame_files]
        
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

@app.route('/get_overlay_frames/<video_name>')  # Added overlay frames list route
def get_overlay_frames_list(video_name):
    overlay_dir = os.path.join(app.config['STATIC_OVERLAY_FOLDER'], video_name)
    
    if os.path.exists(overlay_dir):
        frame_files = sorted([f for f in os.listdir(overlay_dir) if f.endswith('.jpg')])
        frame_urls = [f'/overlay_frames/{video_name}/{f}' for f in frame_files]
        
        return jsonify({
            'success': True,
            'overlay_frames': frame_urls,
            'count': len(frame_files)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Overlay frames directory not found'
        })

@app.route('/get_available_models')
def get_available_models():
    try:
        models_dir = os.path.join('models')
        available_models = []
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pt'):
                    file_path = os.path.join(models_dir, file)
                    file_size = os.path.getsize(file_path)
                    available_models.append({
                        'filename': file,
                        'display_name': file.replace('_', ' ').replace('.pt', '').title(),
                        'size_mb': round(file_size / (1024 * 1024), 1)
                    })
        
        available_models.sort(key=lambda x: x['filename'])
        
        return jsonify({
            'success': True,
            'models': available_models
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reanalyze_audio', methods=['POST'])
def reanalyze_audio():
    try:
        audio_path = request.form.get('audio_path')
        selected_model = request.form.get('selected_model')
        
        if not audio_path or not selected_model:
            return jsonify({'success': False, 'error': 'Audio path and model selection required'})
        
        if not os.path.exists(audio_path):
            return jsonify({'success': False, 'error': 'Audio file not found'})
        
        if not is_audio_detection_available():
            return jsonify({
                'success': False, 
                'error': 'Audio deepfake detection not available. Install PyTorch and torchaudio: pip install torch torchaudio'
            })
        
        models_dir = os.path.join('models')
        model_path = os.path.join(models_dir, selected_model)
        
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': f'Model file not found: {selected_model}'})
        
        print(f"Reanalyzing audio: {audio_path} with model: {selected_model}")
        
        audio_result = detect_audio_deepfake_rawnet(audio_path, model_path)
        audio_result['model_used'] = selected_model
        
        print("Audio reanalysis completed")
        
        return jsonify({
            'success': True,
            'audio_analysis': audio_result
        })
        
    except Exception as e:
        print(f"Audio reanalysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Minimal Deepfake Detection App...")
    print("Open: http://localhost:5000")
    print("Note: Processing timeout is set to 5 minutes for complex videos")
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.timeout = 300
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)