import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Branding based on your Research Paper
st.title("🛡️ FacialOcclusionNet: Real-Time")
st.write("Live Research Demo: MobileNetV1 + Heuristics")

class FacialOcclusionNet(VideoTransformerBase):
    def __init__(self):
        # 1. Load optimized model
        self.interpreter = tf.lite.Interpreter(model_path="mask_model_mobile.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 2. Setup Mediapipe (More robust for mobile than Haar)
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect Faces
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                
                # Preprocess for MobileNetV1 (150x150)
                face_roi = img[max(0,y):y+bh, max(0,x):x+bw]
                if face_roi.size > 0:
                    input_roi = cv2.resize(face_roi, (150, 150))
                    input_roi = np.expand_dims(input_roi.astype('float32') / 255.0, axis=0)
                    
                    # Inference
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_roi)
                    self.interpreter.invoke()
                    pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                    
                    # Heuristics: Texture analysis for N95 detection
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                    
                    # Classification Logic from your script
                    if pred < 0.5:
                        label = "N95 Mask" if variance > 35 else "Surgical Mask"
                        color = (0, 255, 0)
                    else:
                        label = "No Mask"
                        color = (0, 0, 255)

                    cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img

# Start Real-Time Stream
webrtc_streamer(key="facial-occlusion-net", video_transformer_factory=FacialOcclusionNet)