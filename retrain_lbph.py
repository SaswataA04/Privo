"""
Privacy Guard - Dataset Verification and LBPH Retraining

Run from any directory:
    python retrain_lbph.py

This script:
- Verifies the local dataset/saswata images.
- Reports unreadable files, image sizes, and basic brightness statistics.
- Trains the same single-authorized-user LBPH model used by camera_worker.py.
- Backs up an existing privacy_guard_model.yml before replacing it.

It does NOT modify camera_worker.py, settings, or any dataset images.
"""

from pathlib import Path
from datetime import datetime
import shutil
import sys

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "saswata"
MODEL_PATH = PROJECT_DIR / "privacy_guard_model.yml"
BACKUP_DIR = PROJECT_DIR / "model_backups"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FACE_SIZE = (200, 200)
MIN_IMAGES = 20


def main():
    print("=" * 68)
    print("PRIVACY GUARD - DATASET VERIFICATION + LBPH RETRAINING")
    print("=" * 68)

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print("ERROR: OpenCV face module is unavailable.")
        print("Install the contrib build in this virtual environment:")
        print("  pip uninstall opencv-python opencv-python-headless -y")
        print("  pip install opencv-contrib-python")
        return 1

    if not DATASET_DIR.is_dir():
        print(f"ERROR: Dataset folder not found: {DATASET_DIR}")
        return 1

    files = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Dataset: {DATASET_DIR}")
    print(f"Image files found: {len(files)}")

    if len(files) < MIN_IMAGES:
        print(f"ERROR: Found only {len(files)} images; collect at least {MIN_IMAGES} first.")
        return 1

    training_faces = []
    brightness_values = []
    unreadable = []
    dimensions = {}

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            unreadable.append(path.name)
            continue

        dimensions[image.shape] = dimensions.get(image.shape, 0) + 1
        brightness_values.append(float(image.mean()))

        # Match camera_worker.py's LBPH training preprocessing.
        image = cv2.resize(image, FACE_SIZE, interpolation=cv2.INTER_AREA)
        image = cv2.equalizeHist(image)
        training_faces.append(image)

    print(f"Readable images: {len(training_faces)}")
    print(f"Unreadable images: {len(unreadable)}")
    print(f"Original image dimensions: {dimensions}")

    if brightness_values:
        print(
            "Mean brightness range: "
            f"{min(brightness_values):.1f} - {max(brightness_values):.1f}"
        )
        print(f"Average brightness: {np.mean(brightness_values):.1f}")

    if unreadable:
        print("Unreadable files:")
        for name in unreadable:
            print(f"  - {name}")

    if len(training_faces) < MIN_IMAGES:
        print("ERROR: Not enough readable images to train safely.")
        return 1

    print("\nDataset verification passed. Training LBPH model...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.zeros(len(training_faces), dtype=np.int32)
    recognizer.train(training_faces, labels)

    # Preserve the previous model before replacing it.
    if MODEL_PATH.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"privacy_guard_model_{timestamp}.yml"
        shutil.copy2(MODEL_PATH, backup_path)
        print(f"Previous model backed up to: {backup_path}")

    temporary_model = MODEL_PATH.with_name("privacy_guard_model_new.yml")
    recognizer.save(str(temporary_model))
    temporary_model.replace(MODEL_PATH)

    # Re-open the saved model to verify that it can be loaded.
    verification_model = cv2.face.LBPHFaceRecognizer_create()
    verification_model.read(str(MODEL_PATH))

    print("\nSUCCESS: LBPH model trained and saved.")
    print(f"Training images: {len(training_faces)}")
    print(f"Model path: {MODEL_PATH}")
    print("The old model backup, if one existed, is preserved in model_backups/.")
    print("\nNext: restart Privacy Guard and test recognition under the same lighting.")
    print("Note: LBPH confidence is a distance (lower is generally a closer match).")
    print("Do not change the confidence threshold until we inspect actual test results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
