import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import os
import pygame
import torch
from PIL import Image
import time
import tempfile
pygame.mixer.init()

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Text-to-speech initialization
# engine = pyttsx3.init()

# Function to perform object detection and return frame with bounding boxes
def detect_objects(frame, model):
    # Convert the frame to PIL Image for YOLO
    img = Image.fromarray(frame)
    
    # Perform inference
    results = model(img)
    
    # Get detected objects
    detected_objects = results.pandas().xyxy[0]
    detected_labels = set()
    for _, row in detected_objects.iterrows():
        # Draw bounding boxes
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detected_labels.add(label)
        # Speak object name
        # engine.say(label)
        # engine.runAndWait()
    
    if detected_labels:
        text_to_speak = ", ".join(detected_labels)
        tts = gTTS(text=f"Detected objects: {text_to_speak}", lang='en')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file = temp_audio.name
            tts.save(audio_file)
        
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # Wait for the audio to finish
        # os.remove(audio_file)
    
    return frame

# Main Streamlit app
st.title("Real-Time Object Detection with Voice Output")
st.text("Access your device camera to perform real-time object detection.")

# Start video capture
model = load_model()
cap = cv2.VideoCapture(0)  # 0 for default camera

# Streamlit video feed
frame_placeholder = st.empty()

if st.button("Start Detection"):
    last_capture_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access the camera. Please ensure it's connected and accessible.")
            break

        current_time = time.time()
        # Process a frame only every 2 seconds
        if current_time - last_capture_time >= 3:
            last_capture_time = current_time

            # Perform object detection
            frame = detect_objects(frame, model)
            
            # Convert frame to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)


if st.button("Stop Detection"):
    cap.release()
    cv2.destroyAllWindows()
