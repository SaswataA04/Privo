
"""
Privacy Guard — Milestone 2
Face Enrollment

Purpose:
Capture face samples of the authorized user.

Controls:
Q : Quit enrollment
"""

import cv2
import os
import time


# ==========================================================
# CONFIGURATION
# ==========================================================

USER_NAME = "saswata"

DATASET_PATH = os.path.join(
    "dataset",
    USER_NAME
)

TARGET_SAMPLES = 60

# Create directory if it does not exist.
os.makedirs(DATASET_PATH, exist_ok=True)


# ==========================================================
# LOAD FACE DETECTOR
# ==========================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Face detector could not be loaded.")


# ==========================================================
# OPEN WEBCAM
# ==========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to access webcam.")


# ==========================================================
# CAPTURE FACE SAMPLES
# ==========================================================

sample_count = 0

last_capture_time = 0

print("Face enrollment started.")
print("Look at the camera.")
print("Slowly change your head position.")
print("Press Q to stop.")

try:

    while sample_count < TARGET_SAMPLES:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # Capture only when exactly one face is visible.
        if len(faces) == 1:

            x, y, w, h = faces[0]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            current_time = time.time()

            # Capture one sample every 0.2 seconds.
            if current_time - last_capture_time >= 0.2:

                face = gray[y:y+h, x:x+w]

                face = cv2.resize(
                    face,
                    (200, 200)
                )

                filename = os.path.join(
                    DATASET_PATH,
                    f"face_{sample_count}.jpg"
                )

                cv2.imwrite(filename, face)

                sample_count += 1

                last_capture_time = current_time

        # Display progress.
        cv2.putText(
            frame,
            f"Samples: {sample_count}/{TARGET_SAMPLES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Privacy Guard - Face Enrollment",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print(f"Enrollment completed: {sample_count} samples.")
    print(f"Saved in: {DATASET_PATH}")
