<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=4285F4&center=true&vCenter=true&width=500&lines=FacialOcclusionNet:+Real-Time+Face+Mask+Detection;IEEE+Xplore+Research+Publication;99.34%+Accuracy+%7C+MobileNetV2;Streamlit+Edge+Deployment" alt="Typing SVG" />

  <h1>FacialOcclusionNet (FONet)</h1>

  <p>
    <strong>A Novel Real-Time Face Mask Detection Model designed for lightweight edge deployment and robust occlusion handling.</strong>
  </p>

  <p>
    <a href="https://mask-detection-research-lry6ystaf9q3drfgq3raas.streamlit.app/"><strong>Explore the Live Research Demo »</strong></a>
  </p>

  [![IEEE Xplore](https://img.shields.io/badge/Publication-IEEE_Xplore-00629B?style=for-the-badge)](https://ieeexplore.ieee.org/)
  [![Live Demo](https://img.shields.io/badge/Demo-Live_on_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://mask-detection-research-lry6ystaf9q3drfgq3raas.streamlit.app/)
  [![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
</div>

---

## Research Summary
**Lead Author:** Anurag Gaonkar
**Publication:** FacialOcclusionNet: A Novel Real Time Face Mask Detection Model

FacialOcclusionNet (FONet) addresses the critical need for automated health compliance monitoring in high-traffic environments like airports and hospitals. While traditional models struggle with occlusions (e.g., hands covering the face), FONet utilizes a fine-tuned MobileNet architecture to achieve high-speed inference without sacrificing accuracy.

### Key Performance Metrics
* **Test Accuracy:** 99.34%
* **Precision:** 99.73%
* **Recall:** 98.95%
* **F1-Score:** 99.33%

---

## Technical Architecture
The model utilizes transfer learning with a **MobileNetV2** backbone, optimized for edge devices.

* **Preprocessing:** Automated $150 \times 150$ pixel resizing, grayscale normalization, and extensive data augmentation including rotation, zoom, and horizontal flipping.
* **Architecture Highlights:**
    * Frozen base layers from ImageNet to retain robust low-level feature extraction.
    * Custom top layers including Global Average Pooling 2D, a 128-neuron Dense layer with ReLU activation, and a 50% Dropout layer for regularization.
    * Final Sigmoid activation for robust binary classification.
* **Occlusion Handling:** Specifically trained to distinguish between medical masks and non-mask occlusions such as hands or clothing.

---

## Deployment and Usage

### Live Demo
The project is deployed via Streamlit, utilizing the **MediaPipe Tasks API** for real-time face and hand landmark detection to provide an interactive diagnostic experience.

### Local Setup
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/AnuragGaonkar/mask-detection-research.git](https://github.com/AnuragGaonkar/mask-detection-research.git)
   cd mask-detection-research
   ```

### 2. Environment Setup

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Application:**
```bash
streamlit run app.py
```

## Engineering Challenges and Resolutions

### Real-Time Inference on Low-Power Devices
* **Challenge**: High-accuracy CNNs are often computationally heavy, making them unsuitable for real-time monitoring on resource-constrained edge devices.
* **Resolution**: Implemented a lightweight MobileNetV2 backbone with optimized feature extraction, significantly outperforming custom CNN baselines in both speed and accuracy.

### Occlusion Misclassification
* **Challenge**: Standard face mask detectors often trigger false positives when a user covers their face with their hand or clothing.
* **Resolution**: Engineered a two-stage training process consisting of initial feature extraction followed by fine-tuning of deeper layers, combined with hand-landmark detection logic to minimize false positives under dynamic conditions.

---

## Visuals and Demos

### Model Architecture

<img src="research.png" alt="FONet Architecture" width="100%"/>

### Research Walkthrough
*Demonstration of real-time detection under varying lighting and occlusion scenarios.*

> [!IMPORTANT]
> **[Watch High-Resolution Research Demo](research.mp4)**

---

## Contact and Links
**Anurag Gaonkar** - [GitHub](https://github.com/AnuragGaonkar) | [LinkedIn](https://www.linkedin.com/in/anurag-gaonkar-68a463261)

Project Link: [https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION](https://github.com/AnuragGaonkar/STEPPING-STONE-SOLUTION)
