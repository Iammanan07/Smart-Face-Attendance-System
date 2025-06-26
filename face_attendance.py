import pandas as pd
import os
import datetime
from PIL import Image
import numpy as np
import streamlit as st
from data_paths import TRAINING_LABEL_PATH, HAAR_CASCADE_PATH, STUDENT_DETAILS_PATH, ATTENDANCE_DIR
import cv2

def take_attendance(subject, uploaded_image):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINING_LABEL_PATH)
    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    df = pd.read_csv(STUDENT_DETAILS_PATH)
    attendance = []

    # Convert uploaded image to OpenCV format
    img = Image.open(uploaded_image)
    img = img.convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 70:
            matched = df[df['Enrollment'] == id_]
            if matched.empty:
                continue  # unknown ID, skip
            name = matched['Name'].values[0]
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance.append((id_, name, timestamp))

    df_attendance = pd.DataFrame(attendance, columns=["Enrollment", "Name", "Time"])
    df_attendance.drop_duplicates(subset='Enrollment', inplace=True)
    os.makedirs(os.path.join(ATTENDANCE_DIR, subject), exist_ok=True)
    filename = os.path.join(ATTENDANCE_DIR, subject, f"{subject}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    df_attendance.to_csv(filename, index=False)
    return filename
