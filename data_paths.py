import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_IMAGE_PATH = os.path.join(BASE_DIR, "TrainingImage")
TRAINING_LABEL_PATH = os.path.join(BASE_DIR, "TrainingImageLabel", "Trainner.yml")
STUDENT_DETAILS_PATH = os.path.join(BASE_DIR, "StudentDetails", "studentdetails.csv")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

for path in [TRAINING_IMAGE_PATH, os.path.dirname(TRAINING_LABEL_PATH),
             os.path.dirname(STUDENT_DETAILS_PATH), ATTENDANCE_DIR]:
    os.makedirs(path, exist_ok=True)