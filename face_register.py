import cv2
import os

def capture_face_images(enrollment, name, save_dir):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise Exception("Could not open webcam. Please ensure it is connected and accessible.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    folder = os.path.join(save_dir, f"{enrollment}_{name}")
    os.makedirs(folder, exist_ok=True)

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(folder, f"{name}_{enrollment}_{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Faces - Press Q to Stop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    return count
