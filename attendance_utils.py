import os
import pandas as pd
from data_paths import ATTENDANCE_DIR

def list_attendance_files(subject):
    subject_path = os.path.join(ATTENDANCE_DIR, subject)
    if not os.path.exists(subject_path):
        return []
    return [os.path.join(subject_path, f) for f in os.listdir(subject_path) if f.endswith(".csv")]

def read_attendance_file(filepath):
    return pd.read_csv(filepath)