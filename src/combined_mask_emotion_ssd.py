import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import imutils

# ========== LOAD FACE DETECTOR (OpenCV SSD) ==========
faceProto = os.path.sep.join(["face_detector", "deploy.prototxt"])
faceModel = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(faceProto, faceModel)

# ========== LOAD MASK DETECTOR ==========
mask_model = load_model("mask_detector.h5")

# ========== LOAD EMOTION DETECTION MODEL ==========
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

emotion_model = Sequential()
emotion_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, (3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights("model.weights.h5")

# ========== LABELS ==========
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# ========== VIDEO STREAM ==========
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # DNN face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            # ========== MASK DETECTION ==========
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224))
            face_rgb = img_to_array(face_rgb)
            face_rgb = preprocess_input(face_rgb)
            face_rgb = np.expand_dims(face_rgb, axis=0)

            (mask, no_mask) = mask_model.predict(face_rgb)[0]
            mask_label = "Mask" if mask > no_mask else "No Mask"
            color = (0, 255, 0) if mask_label == "Mask" else (0, 0, 255)
            label = f"{mask_label}: {max(mask, no_mask)*100:.2f}%"

            # ========== EMOTION DETECTION ==========
            if mask_label == "No Mask":
                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = gray_face.astype("float") / 255.0
                gray_face = np.expand_dims(gray_face, axis=0)
                gray_face = np.expand_dims(gray_face, axis=-1)

                emotion_pred = emotion_model.predict(gray_face)[0]
                emotion_label = emotion_labels[np.argmax(emotion_pred)]
                label += f" | {emotion_label}"

            # ========== DRAW ==========
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Mask + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
