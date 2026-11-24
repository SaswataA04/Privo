"""
Privacy Guard - Face Enrollment Utility
-----------------------------------
Collects a new, local-only dataset for the authorized user.


IMPORTANT:
- This script does NOT modify camera_worker.py or the recognition model.
- It does NOT delete existing enrollment images.
- It saves normalized grayscale face crops in dataset/saswata.
- Capture is manual (SPACE), so you can deliberately vary your pose.

Controls:
    SPACE   Capture the currently detected face (if quality checks pass)
    BACKSPACE  Delete the most recently captured image from this session
    Q / ESC   Quit enrollment
"""

import cv2
import time
import os
from pathlib import Path



# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------
# Resolve paths relative to this script, so the program works regardless of the
# terminal's current working directory.
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "saswata"
DATASET_DIR.mkdir(parents=True, exist_ok=True)



# -----------------------------------------------------------------------------
# ENROLLMENT PLAN
# -----------------------------------------------------------------------------
# Use these prompts to deliberately capture variety. Press SPACE for each image.
# These are guidance labels only; all images are saved in the same folder because
# the existing LBPH training pipeline expects one folder per authorized person.
# -----------------------------------------------------------------------------
CAPTURE_STAGES = [
    ("LOOK STRAIGHT AT THE CAMERA", 10),
    ("TURN SLIGHTLY LEFT", 10),
    ("TURN SLIGHTLY RIGHT", 10),
    ("LOOK SLIGHTLY UP", 5),
    ("LOOK SLIGHTLY DOWN", 5),
    ("VARY LIGHTING / EXPRESSION / DISTANCE", 10),
]
TOTAL_TARGET = sum(count for _, count in CAPTURE_STAGES)



# -----------------------------------------------------------------------------
# CAPTURE / IMAGE QUALITY SETTINGS
# -----------------------------------------------------------------------------
CAMERA_INDEX = 0
FACE_SIZE = (200, 200)          # Match the current LBPH training size
MIN_FACE_WIDTH = 100            # Reject faces that are too small in the frame
MIN_FACE_HEIGHT = 100
MIN_BRIGHTNESS = 35              # Reject very dark face crops
MAX_BRIGHTNESS = 225             # Reject severely overexposed face crops
CAPTURE_COOLDOWN_SECONDS = 0.45  # Prevent rapid duplicate captures



# -----------------------------------------------------------------------------
# FACE DETECTOR
# -----------------------------------------------------------------------------
# OpenCV's frontal-face Haar cascade is used for face detection. The enrollment
# process does not perform identity recognition or upload any image.
# -----------------------------------------------------------------------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    raise RuntimeError("Could not load OpenCV frontal-face cascade.")



# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def detect_faces(gray_frame):
    """Detect faces and return boxes sorted largest-first."""
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT),
    )
    return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)



def normalize_face(gray_frame, face_box):
    """Crop, resize, and histogram-equalize a face to match LBPH training."""
    x, y, width, height = face_box



    # Add a modest margin around the face to retain useful facial context.
    margin_x = int(width * 0.12)
    margin_y = int(height * 0.12)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(gray_frame.shape[1], x + width + margin_x)
    y2 = min(gray_frame.shape[0], y + height + margin_y)



    crop = gray_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None



    # Consistent crop dimensions are important for the existing LBPH model.
    resized = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(resized)
    return normalized



def brightness_is_acceptable(face_crop):
    """Basic brightness check; it cannot guarantee a perfect image."""
    if face_crop is None or face_crop.size == 0:
        return False, 0.0
    brightness = float(face_crop.mean())
    return MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS, brightness



def current_stage(captured_count):
    """Return the current pose prompt and stage index based on progress."""
    total_so_far = 0
    for index, (prompt, target) in enumerate(CAPTURE_STAGES):
        if captured_count < total_so_far + target:
            return index, prompt, target, captured_count - total_so_far
        total_so_far += target
    return len(CAPTURE_STAGES) - 1, "ENROLLMENT TARGET REACHED - VARY POSES IF DESIRED", CAPTURE_STAGES[-1][1], CAPTURE_STAGES[-1][1]



def get_saved_images():
    """List enrollment image files currently present in the target folder."""
    supported = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted(
        [p for p in DATASET_DIR.iterdir() if p.is_file() and p.suffix.lower() in supported],
        key=lambda p: p.name,
    )



# -----------------------------------------------------------------------------
# MAIN ENROLLMENT LOOP
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("PRIVACY GUARD - FACE ENROLLMENT")
    print("=" * 65)
    print(f"Dataset folder: {DATASET_DIR}")
    print("Images are stored locally. Nothing is uploaded.")
    print("Controls: SPACE=capture | BACKSPACE=undo last | Q/ESC=quit")
    print("Suggested target: about 50 images across varied poses/lighting.")
    print("=" * 65)



    # Open webcam.
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        # Fallback backend on systems where DirectShow isn't available.
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions or index.")
        return



    # Request a useful webcam resolution. The camera may choose a different size.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



    # Files existing before this run are preserved and are not eligible for
    # Backspace removal. Only paths captured during this session are tracked.
    session_files = []
    capture_count = len(get_saved_images())
    last_capture_time = 0.0
    window_name = "Privacy Guard - Face Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)



    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Webcam frame could not be read. Retrying...")
                time.sleep(0.05)
                continue



            # Mirror the webcam preview so movement feels natural to the user.
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            # The largest detected face is selected as the enrollment subject.
            # If multiple faces are visible, enrollment is paused to avoid
            # accidentally collecting another person's face.
            selected_face = None
            if len(faces) == 1:
                selected_face = faces[0]
            elif len(faces) > 1:
                selected_face = None



            # Draw detector results and provide feedback.
            for (x, y, width, height) in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

            if len(faces) == 0:
                face_status = "NO FACE DETECTED - move into frame"
            elif len(faces) > 1:
                face_status = "MULTIPLE FACES - only you should be in frame"
            else:
                face_status = "FACE READY - press SPACE to capture"



            # Stage prompt is based on the number of captured images in this run.
            # Existing images in the folder count toward the overall target.
            stage_index, stage_prompt, stage_target, stage_progress = current_stage(capture_count)
            if capture_count >= TOTAL_TARGET:
                stage_prompt = "TARGET REACHED - add extra varied images if needed"



            # Face brightness is checked before a capture. Use the normalized
            # crop so the check is consistent with what will actually be saved.
            normalized_face = None
            brightness = 0.0
            brightness_ok = False
            if selected_face is not None:
                normalized_face = normalize_face(gray, selected_face)
                brightness_ok, brightness = brightness_is_acceptable(normalized_face)



            # UI overlays.
            cv2.putText(frame, "PRIVACY GUARD | FACE ENROLLMENT", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, face_status, (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 255, 0) if len(faces) == 1 else (0, 165, 255), 2)
            cv2.putText(frame, f"Captured: {capture_count} | Session: {len(session_files)} | Target: {TOTAL_TARGET}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(frame, stage_prompt, (20, 155), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 220, 0), 2)
            cv2.putText(frame, f"Brightness: {brightness:.0f} (acceptable: {MIN_BRIGHTNESS}-{MAX_BRIGHTNESS})",
                        (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if brightness_ok else (0, 165, 255), 2)
            cv2.putText(frame, "SPACE Capture | BACKSPACE Undo | Q/Esc Quit",
                        (20, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF



            # -----------------------------------------------------------------
            # CAPTURE
            # -----------------------------------------------------------------
            if key == ord(' '):
                now = time.monotonic()
                if now - last_capture_time < CAPTURE_COOLDOWN_SECONDS:
                    continue
                if selected_face is None:
                    print("Capture skipped: ensure exactly one face is visible.")
                    continue
                if normalized_face is None:
                    print("Capture skipped: could not crop face.")
                    continue
                if not brightness_ok:
                    print(f"Capture skipped: brightness ({brightness:.0f}) is outside the accepted range.")
                    continue

                # Use a unique, sequential filename without overwriting files.
                existing_numbers = []
                for path in get_saved_images():
                    stem = path.stem
                    if stem.startswith("face_") and stem[5:].isdigit():
                        existing_numbers.append(int(stem[5:]))
                next_number = max(existing_numbers, default=0) + 1
                output_path = DATASET_DIR / f"face_{next_number:04d}.png"
                if cv2.imwrite(str(output_path), normalized_face):
                    session_files.append(output_path)
                    capture_count += 1
                    last_capture_time = now
                    print(f"Saved: {output_path.name} ({capture_count}/{TOTAL_TARGET}) | brightness={brightness:.0f}")
                else:
                    print("ERROR: Failed to save face image.")



            # -----------------------------------------------------------------
            # UNDO LAST CAPTURE (only this session)
            # -----------------------------------------------------------------
            elif key == 8 or key == 127:  # Backspace on common OpenCV key codes
                if session_files:
                    last_file = session_files.pop()
                    try:
                        last_file.unlink(missing_ok=True)
                        capture_count = max(0, capture_count - 1)
                        print(f"Removed last capture: {last_file.name} ({capture_count} total)")
                    except OSError as error:
                        print(f"Could not remove {last_file.name}: {error}")
                else:
                    print("Undo unavailable: no images captured in this session.")

            # Quit enrollment without changing any previously saved images.
            elif key in (ord('q'), 27):
                break


    finally:
        camera.release()
        cv2.destroyAllWindows()



    print("\nEnrollment session ended.")
    print(f"Images in dataset: {len(get_saved_images())}")
    print(f"New images captured this session: {len(session_files)}")
    print(f"Dataset location: {DATASET_DIR}")
    print("Next step after capture: retrain the LBPH model from this dataset.")



if __name__ == "__main__":
    main()
