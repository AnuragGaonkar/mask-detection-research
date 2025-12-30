import os
import cv2
import numpy as np
import tensorflow as tf
import keras 
import mediapipe as mp
import streamlit as st

# ============================================================
# APP CONFIG & MODEL LOADING
# ============================================================
st.set_page_config(page_title="FacialOcclusionNet Demo", layout="centered")
st.title("FacialOcclusionNet")
st.caption("Proof of Work: Mask Detection for Low-End Devices")

@st.cache_resource
def load_model():
    # We use the TFLite version we generated earlier for mobile speed
    interpreter = tf.lite.Interpreter(model_path="mask_model_mobile.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ============================================================
# REFINED HEURISTICS (FROM YOUR CODE)
# ============================================================

def get_texture_variance(region):
    if region is None or region.size == 0: return 0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def has_n95_characteristics(region, texture_var):
    if region is None or region.size == 0: return False
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 130]) 
    upper_white = np.array([180, 65, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(white_mask == 255) / float(region.size/3)
    return white_ratio > 0.55 and texture_var > 30

def has_surgical_characteristics(region):
    if region is None or region.size == 0: return False
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_med = np.array([75, 40, 40]) 
    upper_med = np.array([140, 255, 255])
    med_mask = cv2.inRange(hsv, lower_med, upper_med)
    med_ratio = np.sum(med_mask == 255) / float(region.size/3)
    return med_ratio > 0.30

# ============================================================
# DEPLOYMENT INTERFACE
# ============================================================
img_file = st.camera_input("Scan for FacialOcclusionNet Analysis")

if img_file:
    # 1. Image Decode
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # 2. Deep Learning Classification (TFLite Inference)
    resized = cv2.resize(frame, (150, 150))
    inp = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    is_masked = prediction < 0.5

    # 3. Apply Heuristics (Lower Half Analysis)
    h, w, _ = frame.shape
    lower_half = frame[h//2:h, :]
    
    texture_var = get_texture_variance(lower_half)
    n95_detected = has_n95_characteristics(lower_half, texture_var)
    surg_detected = has_surgical_characteristics(lower_half)

    # 4. Final Classification Logic
    if not is_masked:
        label, color = "No Mask", "red"
    else:
        if n95_detected:
            label, color = "N95 Mask", "green"
        elif surg_detected:
            label, color = "Surgical Mask", "blue" # (Shown as Cyan in UI)
        elif texture_var > 18:
            label, color = "Local/Cloth Mask", "gray" # Subtle dark color
        else:
            label, color = "No Mask (Occlusion)", "orange"

    st.subheader(f"Result: :{color}[{label}]")
    st.write(f"Texture Variance: {texture_var:.2f}")