import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
from streamlit_option_menu import option_menu
from face_train import train_model
from face_attendance import take_attendance
from data_paths import STUDENT_DETAILS_PATH, ATTENDANCE_DIR, TRAINING_IMAGE_PATH

# Page config
st.set_page_config(page_title="Smart Attendance", page_icon="🎓", layout="wide")

# Inject Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #121212, #2c2c2c);
        color: #ffffff;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5em 1em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #b20710;
    }
    .metric-label, .metric-container { color: white; }
    .stTextInput>div>div>input, .stFileUploader>div>div {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #333;
        border-radius: 6px;
    }
    .card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Top Navigation Bar
selected = option_menu(
    menu_title=None,
    options=["Home", "Register", "Train", "Attendance", "Records"],
    icons=["house-fill", "person-plus-fill", "cpu-fill", "camera-fill", "folder-fill"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#141414"},
        "nav-link": {"font-size": "16px", "color": "#fff", "margin":"0 10px"},
        "nav-link-selected": {"background-color": "#e50914"},
    }
)

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
            if os.path.isdir(sub_path):
                count += len(os.listdir(sub_path))
    return count

# HOME
if selected == "Home":
    st.title("📊 Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 Registered Students", get_total_students())
    with col2:
        st.metric("🗂️ Attendance Records", get_total_attendance_records())

    st.markdown("### 👇 Get Started")
    st.markdown("""
    - 📍 Use **Register** to enroll new students.
    - 🧠 Head to **Train** after adding students.
    - 📷 Use **Attendance** to mark presence with a photo.
    - 📁 Visit **Records** to view logs.
    """)

# REGISTER
elif selected == "Register":
    st.title("📝 Register New Student")
    with st.form("register_form"):
        name = st.text_input("Full Name")
        enrollment = st.text_input("Enrollment Number")
        uploaded_file = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])
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
                    st.success(f"✅ {name} ({enrollment}) registered!")
            else:
                st.warning("⚠️ Please fill all fields and upload an image.")

# TRAIN
elif selected == "Train":
    st.title("🧠 Train Face Recognition Model")
    st.write("Click the button below to train your model.")
    if st.button("🛠️ Start Training"):
        count = train_model()
        if count:
            st.success(f"✅ Model trained on {count} students.")
        else:
            st.warning("⚠️ No faces found to train.")

# ATTENDANCE
elif selected == "Attendance":
    st.title("📷 Mark Attendance")
    subject = st.text_input("Enter Subject Name")
    uploaded_image = st.file_uploader("Upload Attendance Image", type=["jpg", "jpeg", "png"])
    if uploaded_image and subject:
        file = take_attendance(subject, uploaded_image)
        st.success(f"✅ Attendance saved to: {file}")
        df = pd.read_csv(file)
        st.dataframe(df)

# RECORDS
elif selected == "Records":
    st.title("📁 Attendance Records")
    subject = st.text_input("Enter Subject")
    if subject:
        subject_path = os.path.join(ATTENDANCE_DIR, subject)
        if os.path.exists(subject_path):
            files = os.listdir(subject_path)
            if files:
                file = st.selectbox("📂 Select Record", files)
                df = pd.read_csv(os.path.join(subject_path, file))
                st.dataframe(df)
                st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=file)
            else:
                st.info("ℹ️ No records found.")
        else:
            st.warning("⚠️ Subject not found.")
