import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
import keras 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Branding from Research Paper
st.set_page_config(page_title="FacialOcclusionNet", layout="centered")
st.title("FacialOcclusionNet")
st.write("Real-Time Research Demo")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class FacialOcclusionProcessor:
    def __init__(self):
        # 1. FIXED: Loading the original high-accuracy model instead of Lite
        # compile=False is used for maximum environment compatibility
        self.model = keras.models.load_model('best_model.keras', compile=False)
        
        # 2. Setup MediaPipe Face Detector (Tasks API)
        face_base_options = python.BaseOptions(model_asset_path='face_detector.tflite')
        face_options = vision.FaceDetectorOptions(base_options=face_base_options)
        self.detector = vision.FaceDetector.create_from_options(face_options)

        # 3. Setup Hands for Occlusion Logic
        hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # 4. Persistence Counters (Exactly as per your logic)
        self.n95_counter = 0 
        self.surg_counter = 0

    def get_texture_variance(self, region):
        if region is None or region.size == 0: return 0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def has_n95_characteristics(self, region, texture_var):
        if region is None or region.size == 0: return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 130]) 
        upper_white = np.array([180, 65, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_ratio = np.sum(white_mask == 255) / float(region.size/3)
        return white_ratio > 0.55 and texture_var > 30

    def has_surgical_characteristics(self, region):
        if region is None or region.size == 0: return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower_med = np.array([75, 40, 40]) 
        upper_med = np.array([140, 255, 255])
        med_mask = cv2.inRange(hsv, lower_med, upper_med)
        med_ratio = np.sum(med_mask == 255) / float(region.size/3)
        return med_ratio > 0.30

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        face_results = self.detector.detect(mp_image)
        hand_results = self.landmarker.detect(mp_image)

        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.bounding_box
                fx, fy, fw, fh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                fx, fy = max(0, fx), max(0, fy)
                
                face_roi = img[fy:fy+fh, fx:fx+fw]
                if face_roi.size == 0: continue

                # Full accuracy Keras Model Inference
                resized = cv2.resize(face_roi, (150, 150))
                normalized = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
                prediction = float(self.model.predict(normalized, verbose=0)[0][0])
                is_masked_by_model = prediction < 0.5

                # Hand check
                hand_near = False
                if hand_results.hand_landmarks:
                    for hand_lms in hand_results.hand_landmarks:
                        for lm in hand_lms:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            if fx < cx < fx+fw and fy < cy < fy+fh:
                                hand_near = True; break

                lower_y = fy + fh // 2
                lower_roi = img[lower_y:fy+fh, fx:fx+fw]
                
                label = "No Mask"
                color = (0, 0, 255) 

                if is_masked_by_model and lower_roi.size > 0:
                    texture_var = self.get_texture_variance(lower_roi)
                    surg_detected = self.has_surgical_characteristics(lower_roi)
                    n95_detected = self.has_n95_characteristics(lower_roi, texture_var)
                    
                    if n95_detected: self.n95_counter = min(self.n95_counter + 3, 12)
                    else: self.n95_counter = max(self.n95_counter - 1, 0)
                    
                    if surg_detected: self.surg_counter = min(self.surg_counter + 3, 12)
                    else: self.surg_counter = max(self.surg_counter - 1, 0)

                    if hand_near:
                        label, color = "No Mask (Occlusion)", (0, 165, 255)
                    elif self.n95_counter > 4:
                        label, color = "N95 Mask", (0, 255, 0)
                    elif self.surg_counter > 4:
                        label, color = "Surgical Mask", (255, 255, 0)
                    elif texture_var > 18:
                        label, color = "Local/Cloth Mask", (139, 0, 0)
                    else:
                        label, color = "No Mask (Occlusion)", (0, 165, 255)

                cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), color, 2)
                cv2.putText(img, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="FacialOcclusionNet",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=FacialOcclusionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    # REMOVES VIDEO PLAYER CONTROLS
    video_html_attrs={
        "style": {"width": "100%", "border": "2px solid #555"},
        "controls": False,
        "autoPlay": True,
    },
    async_processing=True, # HIGHER FPS
)