import cv2
import os
import pandas as pd
from datetime import datetime
from data_paths import STUDENT_DETAILS_PATH, TRAINING_LABEL_PATH, ATTENDANCE_DIR
import numpy as np
from PIL import Image

def take_attendance(subject, uploaded_image):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINING_LABEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    img = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    df = pd.read_csv(STUDENT_DETAILS_PATH)
    attendance = []

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        name_row = df[df['Enrollment'] == id_]
        if not name_row.empty:
            name = name_row.iloc[0]['Name']
            now = datetime.now().strftime("%H:%M:%S")
            attendance.append({"Enrollment": id_, "Name": name, "Time": now})

    df_att = pd.DataFrame(attendance)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(ATTENDANCE_DIR, subject)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"{date_str}.csv")
    df_att.to_csv(file, index=False)
    return file
