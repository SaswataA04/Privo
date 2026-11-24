"""
Privacy Guard - Automatic Face Enrollment v2
--------------------------------------------
Collects a fresh, local-only face dataset without requiring SPACE for every image.

Run from the project root:
    python enroll_face_auto.py

Output:
    dataset/saswata_auto_v2/

Controls:
    N       Skip to the next guided pose stage
    Q/ESC   Finish and exit

The script does NOT modify camera_worker.py, the existing dataset, or the LBPH model.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np


# ----------------------------- Configuration -----------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset" / "saswata_auto_v2"

IMAGE_SIZE = (200, 200)
TOTAL_TARGET = 60
IMAGES_PER_STAGE = 12
CAPTURE_INTERVAL_SECONDS = 0.50

# Basic quality gates. These are deliberately conservative starting values.
MIN_FACE_SIZE = 90
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
MIN_BLUR_SCORE = 30.0

# Reject near-identical consecutive captures. This is a simple image-difference
# check, not a biometric identity test.
MIN_FRAME_DIFFERENCE = 3.0

STAGES = [
    ("FRONT", "Look directly at the camera. Keep a natural expression."),
    ("SLIGHT LEFT", "Turn your head slightly to your LEFT."),
    ("SLIGHT RIGHT", "Turn your head slightly to your RIGHT."),
    ("SLIGHT UP", "Tilt your chin slightly UP."),
    ("SLIGHT DOWN", "Tilt your chin slightly DOWN."),
]


# ----------------------------- Helper functions -----------------------------

def draw_text(frame, text, position, scale=0.65, color=(255, 255, 255), thickness=2):
    """Draw readable text with a dark outline."""
    x, y = position
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def make_face_crop(gray_frame, face_box):
    """
    Crop a square face region with a modest margin, resize it to 200x200,
    and apply histogram equalization to match the existing LBPH pipeline.
    """
    x, y, w, h = face_box
    center_x = x + w // 2
    center_y = y + h // 2

    # Square crop based on the larger face dimension, with 12% margin.
    side = int(max(w, h) * 1.24)
    half = side // 2

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(gray_frame.shape[1], center_x + half)
    y2 = min(gray_frame.shape[0], center_y + half)

    crop = gray_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    normalized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(normalized)
    return normalized


def image_difference(image_a, image_b):
    """Return mean absolute pixel difference between two normalized crops."""
    if image_a is None or image_b is None:
        return 255.0
    return float(cv2.absdiff(image_a, image_b).mean())


def ensure_safe_output_folder():
    """
    Refuse to overwrite an existing non-empty enrollment folder.
    This protects earlier captures if the script is run again.
    """
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"The output folder already contains files:\n{OUTPUT_DIR}\n\n"
            "Rename that folder or move it aside before starting a new enrollment. "
            "The script will not overwrite existing images."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------- Main flow --------------------------------

def main():
    print("=" * 68)
    print("PRIVACY GUARD - AUTOMATIC FACE ENROLLMENT V2")
    print("=" * 68)
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Target: {TOTAL_TARGET} accepted images")
    print("No images are uploaded. This script does not train or replace a model.")
    print()

    try:
        ensure_safe_output_folder()
    except Exception as exc:
        print(f"SETUP ERROR: {exc}")
        return

    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("ERROR: Could not load OpenCV Haar face detector.")
        return

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam. Close other apps using the camera.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    total_saved = 0
    stage_index = 0
    stage_saved = 0
    last_capture_time = 0.0
    last_saved_crop = None
    rejected_reason = "Position your face inside the frame"

    window_name = "Privacy Guard - Automatic Face Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Enrollment started. Follow the on-screen pose instructions.")
    print("Captures happen automatically; do not press SPACE.")
    print("Press N to skip a pose stage, or Q/ESC to stop.")
    print()

    try:
        while total_saved < TOTAL_TARGET:
            ok, frame = camera.read()
            if not ok:
                print("ERROR: Failed to read a frame from the webcam.")
                break

            # Mirror preview so movements feel natural.
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            current_crop = None
            current_face = None

            if len(faces) == 1:
                current_face = tuple(int(v) for v in faces[0])
                x, y, w, h = current_face
                current_crop = make_face_crop(gray, current_face)

                if current_crop is None:
                    rejected_reason = "Could not create face crop"
                else:
                    brightness = float(np.mean(current_crop))
                    blur_score = float(cv2.Laplacian(current_crop, cv2.CV_64F).var())

                    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
                        rejected_reason = f"Lighting unsuitable ({brightness:.0f})"
                    elif blur_score < MIN_BLUR_SCORE:
                        rejected_reason = f"Hold still - image is blurry ({blur_score:.0f})"
                    else:
                        rejected_reason = "Ready - automatic capture active"

                    color = (0, 220, 0) if rejected_reason.startswith("Ready") else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            elif len(faces) == 0:
                rejected_reason = "No face detected - face the webcam"
            else:
                rejected_reason = "Multiple faces detected - ensure only you are visible"

            now = time.monotonic()
            quality_ok = (
                current_crop is not None
                and rejected_reason.startswith("Ready")
            )

            if quality_ok and now - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
                difference = image_difference(current_crop, last_saved_crop)
                if last_saved_crop is None or difference >= MIN_FRAME_DIFFERENCE:
                    filename = OUTPUT_DIR / f"face_{total_saved + 1:03d}.png"
                    if cv2.imwrite(str(filename), current_crop):
                        total_saved += 1
                        stage_saved += 1
                        last_capture_time = now
                        last_saved_crop = current_crop.copy()
                        print(
                            f"Saved {total_saved:02d}/{TOTAL_TARGET} "
                            f"({STAGES[stage_index][0]}) -> {filename.name}"
                        )
                    else:
                        rejected_reason = "Could not save image - check folder permissions"
                else:
                    rejected_reason = "Skipping near-duplicate frame - move slightly"

            # Advance automatically after the target number for each stage.
            if stage_saved >= IMAGES_PER_STAGE and stage_index < len(STAGES) - 1:
                stage_index += 1
                stage_saved = 0
                last_saved_crop = None
                print(f"\nNext pose: {STAGES[stage_index][0]}\n")

            stage_name, stage_instruction = STAGES[stage_index]
            progress_width = 400
            progress = int(progress_width * total_saved / TOTAL_TARGET)
            cv2.rectangle(frame, (20, 20), (20 + progress_width, 42), (70, 70, 70), -1)
            cv2.rectangle(frame, (20, 20), (20 + progress, 42), (0, 200, 0), -1)

            draw_text(frame, f"ENROLLMENT: {total_saved}/{TOTAL_TARGET}", (20, 75), 0.75)
            draw_text(frame, f"POSE: {stage_name}", (20, 110), 0.7, (0, 230, 255))
            draw_text(frame, stage_instruction, (20, 145), 0.52)
            draw_text(frame, rejected_reason, (20, frame.shape[0] - 55), 0.55,
                      (0, 220, 0) if rejected_reason.startswith("Ready") else (0, 165, 255))
            draw_text(frame, "N: next pose   Q/ESC: finish", (20, frame.shape[0] - 20), 0.48)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                print("Enrollment stopped by user.")
                break
            elif key in (ord("n"), ord("N")):
                if stage_index < len(STAGES) - 1:
                    stage_index += 1
                    stage_saved = 0
                    last_saved_crop = None
                    print(f"Skipped to pose: {STAGES[stage_index][0]}")
                else:
                    print("Already at the final pose stage.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print()
    print("=" * 68)
    print("ENROLLMENT SUMMARY")
    print("=" * 68)
    print(f"Images saved: {total_saved}")
    print(f"Dataset folder: {OUTPUT_DIR}")
    if total_saved == TOTAL_TARGET:
        print("SUCCESS: Target reached.")
    elif total_saved > 0:
        print("PARTIAL: Some images were collected. You can inspect them before continuing.")
    else:
        print("No images were saved. Check camera access and lighting.")
    print()
    print("IMPORTANT: The existing dataset and privacy_guard_model.yml were not changed.")
    print("Do not retrain or replace the working model until the new images are inspected.")


if __name__ == "__main__":
    main()
