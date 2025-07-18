# 😷🧠 Face Mask and Emotion Detection using Deep Learning

A real-time deep learning application that detects if a person is wearing a mask 😷 and identifies their facial emotion 😊😠😢 using computer vision and neural networks.

---

## 🧠 What It Does

- 🔍 Detects human faces in real-time
- 😷 Classifies whether a person is wearing a **Mask / No Mask**
- 😀 Detects facial emotions:
  - Angry
  - Disgusted
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised

---

## 🧰 Tech Stack

| Task                   | Tools & Frameworks                              |
|------------------------|-------------------------------------------------|
| Face Detection         | OpenCV DNN (`deploy.prototxt`, SSD model)       |
| Mask Classification    | MobileNetV2 (TensorFlow/Keras)                  |
| Emotion Classification | Custom CNN (FER-2013 dataset)                   |
| GUI                    | Tkinter (Python GUI)                            |
| Deployment             | PyInstaller (.exe generation)                   |

---
[![Download EXE](https://img.shields.io/badge/Download-EXE-blue?style=for-the-badge&logo=windows)](https://github.com/Tanmay0405/mask-emotion-detector/releases/latest)

## 🚀 How to Run the Project

## 📥 Dataset Download

> Note: Due to GitHub file size limits, the full training datasets are not included in this repo.

- [Download Mask Dataset (RMFD)](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [Download FER-2013 Emotion Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

After downloading:
- Place the mask dataset inside `dataset/`
- Use `data/` for emotion dataset preprocessing


### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt


Note: This project requires Python 3.7+ and TensorFlow 2.x

### 2. 🏃 Run Real-Time Detection

2. 🖥 Run via GUI

```bash

python src/gui_main.py

Or double-click the .exe (if available).


Note: Make sure your webcam is enabled. Press Q to quit the webcam mode application.

3. 💻 Real-Time Detection (Script Mode)

bash

python src/combined_mask_emotion_ssd.py

Press Q to quit the webcam feed.

## 🖥 Executable (Windows)

You can build the `.exe` using:

```bash
python -m PyInstaller --noconfirm --onefile ...



 📁 Folder Structure
MASK_EMOTIONS_DETECTION/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── dist/
│   └── gui_main.exe            # ❌ Not on GitHub
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── models/
│   ├── mask_detector.h5
│   └── model.weights.h5
│
├── screenshots/
│   └── sample1.png
│   └── sample2.png
│
├── src/
│   ├── combined_mask_emotion_ssd.py
│   ├── dataset_prepare.py
│   ├── detect_mask_video.py
│   ├── emotions.py
│   ├── gui_main.py
│   └── train_mask_detector.py
│
├── haarcascade_frontalface_default.xml
├── gui_main.spec
├── requirements.txt
└── README.md



  📊 Model Info
😷 Mask Detection
Model: MobileNetV2

Input: 224x224 RGB

Accuracy: ~95%

  😀 Emotion Detection
Model: Custom CNN

Dataset: FER-2013

Input: 48x48 grayscale

Accuracy: ~65–70%


📸 Screenshots
 <p align="center">
 <img src="screenshots/sample1.png" width="400"/> 
 <img src="screenshots/sample2.png" width="400"/> 
 </p>


📚 Dataset Credits
FER-2013: Kaggle FER Challenge

Mask Dataset: RMFD, Kaggle, and public datasets



💡 Future Improvements
    Add gender and age prediction

    Replace SSD with YOLOv8 for face detection

    Create a Streamlit / Flask web app

    Optimize for mobile using TFLite or ONNX

## 👤 Author

**Tanmay Awasthi**  
🎓 Deep Learning Enthusiast | Final Year B.Tech Student  
🔗 [GitHub @Tanmay0405](https://github.com/Tanmay0405)  
🔗 [LinkedIn](https://www.linkedin.com/in/tanmay-awasthi-programmer4)
