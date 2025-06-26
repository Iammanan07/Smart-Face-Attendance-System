import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
from face_train import train_model
from face_attendance import take_attendance
from attendance_utils import list_attendance_files, read_attendance_file
from data_paths import STUDENT_DETAILS_PATH, ATTENDANCE_DIR, TRAINING_IMAGE_PATH

st.set_page_config(page_title="Smart Attendance System", page_icon="🎓", layout="wide")

# Inject styling
st.markdown("""
<style>
body { background-color: #0f0f0f; color: #f1f1f1; font-family: 'Segoe UI', sans-serif; }
.stApp { background: #1c1c1e; }
h1, h2, h3, h4 { color: #ffcc00; }
.metric-label, .metric-container { color: white; }
.block-container { padding: 2rem; background-color: #1c1c1e; border-radius: 12px; }
section.main > div { padding-top: 2rem; }
.stButton > button {
  background-color: #e50914;
  color: white;
  border-radius: 8px;
  font-size: 16px;
  padding: 0.5rem 1rem;
}
.stButton > button:hover {
  background-color: #c40812;
}
input, .stTextInput input {
  background-color: #2e2e2e;
  color: white;
  border: 1px solid #555;
  border-radius: 6px;
  padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

section = st.sidebar.radio("📂 Navigate to:", ["Home", "Register Student", "Train Model", "Take Attendance", "View Records"])

st.title("🎓 Face Recognition Attendance System")

# Helper functions
def get_total_students():
    if os.path.exists(STUDENT_DETAILS_PATH):
        df = pd.read_csv(STUDENT_DETAILS_PATH)
        return len(df)
    return 0

def get_total_attendance_records():
    count = 0
    if os.path.exists(ATTENDANCE_DIR):
        for subject in os.listdir(ATTENDANCE_DIR):
            sub_path = os.path.join(ATTENDANCE_DIR, subject)
            if os.path.isdir(sub_path):  # ✅ Only count folders
                count += len(os.listdir(sub_path))
    return count


# Home
if section == "Home":
    st.header("📊 Dashboard Overview")
    col1, col2 = st.columns(2)
    col1.metric("👥 Registered Students", get_total_students())
    col2.metric("🗂️ Attendance Records", get_total_attendance_records())
    st.markdown("---")
    st.markdown("Use the side menu to register students, train your model, or track attendance easily.")

# Register Student
elif section == "Register Student":
    st.header("📝 Register New Student")
    with st.form("register_form"):
        name = st.text_input("👤 Full Name")
        enrollment = st.text_input("🎓 Enrollment Number")
        uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("📩 Register")

        if submitted:
            if uploaded_file and name and enrollment:
                folder = os.path.join(TRAINING_IMAGE_PATH, f"{enrollment}_{name}")
                os.makedirs(folder, exist_ok=True)
                img_path = os.path.join(folder, f"{name}_{enrollment}_1.jpg")
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.image(img_path, caption="📸 Uploaded Image", use_column_width=True)

                if os.path.exists(STUDENT_DETAILS_PATH):
                    df = pd.read_csv(STUDENT_DETAILS_PATH)
                else:
                    df = pd.DataFrame(columns=["Enrollment", "Name"])
                if not ((df['Enrollment'] == int(enrollment)) & (df['Name'] == name)).any():
                    df.loc[len(df.index)] = [int(enrollment), name]
                    df.to_csv(STUDENT_DETAILS_PATH, index=False)
                    st.success(f"✅ {name} ({enrollment}) registered successfully!")
            else:
                st.warning("⚠️ Please fill in all fields and upload an image.")

# Train Model
elif section == "Train Model":
    st.header("🧠 Train Face Recognition Model")
    st.write("Train the model using the currently registered student images.")
    if st.button("🛠️ Train Now"):
        count = train_model()
        st.success(f"✅ Model trained on {count} students")

# Take Attendance
elif section == "Take Attendance":
    st.header("📷 Take Attendance")
    subject = st.text_input("📘 Enter Subject Name")
    uploaded_image = st.camera_input("📸 Take a photo to mark attendance")

    if uploaded_image and subject:
        file = take_attendance(subject, uploaded_image)
        st.success(f"✅ Attendance saved to {file}")
        df = pd.read_csv(file)
        st.dataframe(df)

# View Records
elif section == "View Records":
    st.header("📑 View Attendance Records")
    subject = st.text_input("📘 Subject")
    if subject:
        files = list_attendance_files(subject)
        if files:
            file = st.selectbox("📁 Select Record", files)
            df = read_attendance_file(file)
            st.dataframe(df)
            st.download_button("⬇️ Download CSV", data=df.to_csv(index=False), file_name=os.path.basename(file))
        else:
            st.info("ℹ️ No records found for this subject.")
