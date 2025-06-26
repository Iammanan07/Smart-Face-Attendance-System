import os
import numpy as np
import face_recognition
import cv2
from data_paths import TRAINING_IMAGE_PATH, TRAINING_LABEL_PATH

def train_model():
    faces = []
    ids = []
    label_map = {}
    current_id = 0

    for student_folder in os.listdir(TRAINING_IMAGE_PATH):
        path = os.path.join(TRAINING_IMAGE_PATH, student_folder)
        if not os.path.isdir(path):
            continue
        enrollment_id = int(student_folder.split('_')[0])

        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                faces.append(encodings[0])
                ids.append(enrollment_id)

    # Save encodings and labels
    np.savez(TRAINING_LABEL_PATH, encodings=faces, ids=ids)
    return len(set(ids))
