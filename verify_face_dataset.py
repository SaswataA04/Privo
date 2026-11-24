"""
Privacy Guard - Verify automatic face enrollment dataset.
Read-only: does not modify images, datasets, or trained models.

Run:
    python verify_face_dataset.py
"""

from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "saswata_auto_v2"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def main():
    print("=" * 68)
    print("PRIVACY GUARD - AUTOMATIC DATASET VERIFICATION")
    print("=" * 68)
    print(f"Dataset: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder does not exist.")
        return

    files = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    print(f"Image files found: {len(files)}")

    if not files:
        print("ERROR: No supported image files found.")
        return

    readable = 0
    unreadable = []
    dimensions = {}
    brightness_values = []
    blur_values = []

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            unreadable.append(path.name)
            continue

        readable += 1
        dimensions[image.shape[::-1]] = dimensions.get(image.shape[::-1], 0) + 1
        brightness_values.append(float(np.mean(image)))
        blur_values.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))

    print(f"Readable images: {readable}")
    print(f"Unreadable images: {len(unreadable)}")
    print(f"Dimensions: {dimensions}")

    if brightness_values:
        print(
            "Brightness min/max/average: "
            f"{min(brightness_values):.1f} / "
            f"{max(brightness_values):.1f} / "
            f"{np.mean(brightness_values):.1f}"
        )
        print(
            "Blur score min/max/average: "
            f"{min(blur_values):.1f} / "
            f"{max(blur_values):.1f} / "
            f"{np.mean(blur_values):.1f}"
        )

    if unreadable:
        print("Unreadable files:")
        for name in unreadable:
            print(f"  - {name}")

    print("-" * 68)
    if readable == len(files) and len(files) >= 40:
        print("BASIC CHECK PASSED: Dataset is readable and has enough images.")
    else:
        print("CHECK NEEDS ATTENTION: Review image count and unreadable files.")

    print("This is a file/quality check only, not proof of identity or pose diversity.")
    print("No files or models were changed.")

if __name__ == "__main__":
    main()
