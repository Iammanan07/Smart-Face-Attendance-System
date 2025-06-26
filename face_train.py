import os
import cv2
import numpy as np
from data_paths import TRAINING_IMAGE_PATH, TRAINING_LABEL_PATH

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []

    for student_folder in os.listdir(TRAINING_IMAGE_PATH):
        folder_path = os.path.join(TRAINING_IMAGE_PATH, student_folder)
        if not os.path.isdir(folder_path):
            continue

        enrollment_id = int(student_folder.split('_')[0])

        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_samples.append(gray[y:y + h, x:x + w])
                ids.append(enrollment_id)

    if not face_samples:
        return 0

    recognizer.train(face_samples, np.array(ids))
    os.makedirs(os.path.dirname(TRAINING_LABEL_PATH), exist_ok=True)
    recognizer.save(TRAINING_LABEL_PATH)
    return len(set(ids))
