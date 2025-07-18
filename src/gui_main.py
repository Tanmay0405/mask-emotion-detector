import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import sys
import os

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller .exe"""
    try:
        base_path = sys._MEIPASS  # PyInstaller extracts to this at runtime
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Emotion Labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Model paths
FACE_PROTO = resource_path("face_detector/deploy.prototxt")
FACE_MODEL = resource_path("face_detector/res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL = resource_path("models/mask_detector.h5")
EMOTION_MODEL = resource_path("models/model.weights.h5")
HAAR_CASCADE = resource_path("haarcascade_frontalface_default.xml")

# Load face detection model
face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

# Load mask detection model
mask_model = load_model(MASK_MODEL)


# Load emotion model
def load_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights(EMOTION_MODEL)
    return model

emotion_model = load_emotion_model()

# ===========================
# Webcam Detection Function
# ===========================
def detect_from_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                face = frame[y1:y2, x1:x2]

                if face.size > 0:
                    # Mask detection
                    resized = cv2.resize(face, (224, 224))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    tensor = tf.keras.preprocessing.image.img_to_array(rgb)
                    tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
                    tensor = np.expand_dims(tensor, axis=0)
                    (mask, withoutMask) = mask_model.predict(tensor)[0]

                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # Emotion detection
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    try:
                        roi = cv2.resize(gray, (48, 48))
                        roi = roi.astype("float") / 255.0
                        roi = np.reshape(roi, (1, 48, 48, 1))
                        preds = emotion_model.predict(roi)
                        emotion = emotion_dict[np.argmax(preds)]
                    except:
                        emotion = "Unknown"

                    display = f"{label}, {emotion}"
                    cv2.putText(frame, display, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===========================
# Image Detection Function
# ===========================
def detect_from_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not filepath:
        return

    frame = cv2.imread(filepath)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                resized = cv2.resize(face, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = tf.keras.preprocessing.image.img_to_array(rgb)
                tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
                tensor = np.expand_dims(tensor, axis=0)
                (mask, withoutMask) = mask_model.predict(tensor)[0]

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                try:
                    roi = cv2.resize(gray, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = np.reshape(roi, (1, 48, 48, 1))
                    preds = emotion_model.predict(roi)
                    emotion = emotion_dict[np.argmax(preds)]
                except:
                    emotion = "Unknown"

                display = f"{label}, {emotion}"
                cv2.putText(frame, display, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Image Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===========================
# GUI Setup
# ===========================
window = tk.Tk()
window.title("Mask + Emotion Detector")
window.geometry("400x250")
window.configure(bg="#f2f2f2")

tk.Label(window, text="Choose an Option:", font=("Arial", 14), bg="#f2f2f2").pack(pady=20)
tk.Button(window, text="🖼 Detect from Image", font=("Arial", 12), command=detect_from_image, width=25, bg="#4caf50", fg="white").pack(pady=10)
tk.Button(window, text="📷 Detect from Webcam", font=("Arial", 12), command=detect_from_webcam, width=25, bg="#2196f3", fg="white").pack(pady=10)
tk.Label(window, text="Press Q to quit webcam view", font=("Arial", 9), bg="#f2f2f2").pack(pady=10)

window.mainloop()
