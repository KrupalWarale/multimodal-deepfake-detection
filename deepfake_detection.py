#!/usr/bin/env python3
"""
Minimal Deepfake Detection System
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TIME_STEPS = 15
HEIGHT = 224
WIDTH = 224

def build_model(lstm_hidden_size=256, num_classes=2, dropout_rate=0.5):
    """Build MobileNetV2 + LSTM model for deepfake detection"""
    inputs = layers.Input(shape=(TIME_STEPS, HEIGHT, WIDTH, 3))
    
    base_model = keras.applications.MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        pooling='avg',
        input_shape=(HEIGHT, WIDTH, 3)
    )
    base_model.trainable = False
    
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.LSTM(lstm_hidden_size//2)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def detect_deepfake(video_path, model_path=''):
    """Video deepfake detection"""
    try:
        print("Building detection model...")
        import warnings
        warnings.filterwarnings('ignore')
        
        model = build_model()
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from: {model_path}")
            model.load_weights(model_path)
        else:
            print("Using pretrained base model only (no custom weights)")
        
        print("Processing video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_skip = max(1, total_frames // TIME_STEPS) if total_frames > TIME_STEPS else 1
        
        frame_count = 0
        while len(frames) < TIME_STEPS:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                frames.append(frame_normalized)
            
            frame_count += 1
        
        cap.release()
        
        while len(frames) < TIME_STEPS:
            frames.append(frames[-1] if frames else np.zeros((HEIGHT, WIDTH, 3)))
        
        print("Running inference...")
        video_array = np.array([frames[:TIME_STEPS]])
        predictions = model.predict(video_array, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0]
        
        class_names = ['Real', 'Fake']
        print(f"Detection complete: {class_names[predicted_class]} ({probabilities[predicted_class]:.3f})")
        return {
            'predicted_class': class_names[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'probabilities': {'Real': float(probabilities[0]), 'Fake': float(probabilities[1])}
        }
        
    except Exception as e:
        return {'predicted_class': 'Error', 'confidence': 0.0, 'error': str(e)}

def detect_video_only(video_path):
    """Video-only deepfake detection"""
    try:
        video_result = detect_deepfake(video_path)
        return {
            'video_analysis': video_result,
            'audio_analysis': None,
            'combined_verdict': video_result.get('predicted_class', 'Error'),
            'combined_confidence': video_result.get('confidence', 0.0),
            'combined_probabilities': video_result.get('probabilities', {'Real': 0.5, 'Fake': 0.5})
        }
    except Exception as e:
        print(f"Video analysis failed: {e}")
        return {
            'video_analysis': {'predicted_class': 'Error', 'error': str(e)},
            'audio_analysis': None,
            'combined_verdict': 'Error',
            'combined_confidence': 0.0,
            'combined_probabilities': {'Real': 0.5, 'Fake': 0.5}
        }