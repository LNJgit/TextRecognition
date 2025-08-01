import os
import cv2
import matplotlib.pyplot as plt
from face_utils import FaceDetection
from text_utils import TextDetector
from db.db import SessionLocal
from db.operations import save_or_update_ocr_result, find_images_by_word

def process_directory(assets_dir):
    db = SessionLocal()
    # --- Process all images ---
    for filename in os.listdir(assets_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-images

        image_path = os.path.join(assets_dir, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: could not read {image_path}")
            continue

        print(f"Processing {filename}...")

        words = text_detector.DetectAndDisplay(img)
        save_or_update_ocr_result(db, image_filename=filename, words=words)
    db.close()


# --- Setup ---
assets_dir = 'assets'
text_detector = TextDetector(GPU_enabled=True)
process_directory(assets_dir)

# --- Test search ---
db = SessionLocal()
results = find_images_by_word(db, "alphabet")
db.close()

print("Found images:")
for img in results:
    print("-", img.filename)
