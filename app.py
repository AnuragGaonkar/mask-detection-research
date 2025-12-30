import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Branding
st.set_page_config(page_title="FacialOcclusionNet", layout="centered")
st.title("FacialOcclusionNet")
st.write("Real-Time Research Demo: MobileNetV1 + Heuristic Layers")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class FacialOcclusionProcessor:
    def __init__(self):
        # 1. Load your TFLite mask model
        self.interpreter = tf.lite.Interpreter(model_path="mask_model_mobile.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 2. Setup MediaPipe Face Detector (Tasks API)
        # IMPORTANT: You must have 'face_detector.tflite' in your GitHub repo!
        model_path = 'face_detector.tflite'
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
        else:
            self.detector = None
            print("Error: face_detector.tflite not found.")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.detector is None: return frame.from_ndarray(img, format="bgr24")

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detection_result = self.detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                
                # Boundary safety
                x, y = max(0, x), max(0, y)
                face_roi = img[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # MobileNetV1 Inference
                    input_roi = cv2.resize(face_roi, (150, 150))
                    input_roi = np.expand_dims(input_roi.astype('float32') / 255.0, axis=0)
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_roi)
                    self.interpreter.invoke()
                    pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                    
                    # Laplacian Heuristic
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                    
                    # Classification
                    if pred < 0.5:
                        label = "N95 Mask" if variance > 35 else "Surgical Mask"
                        color = (0, 255, 0)
                    else:
                        label, color = "No Mask / Occlusion", (0, 0, 255)

                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="FacialOcclusionNet",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=FacialOcclusionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)