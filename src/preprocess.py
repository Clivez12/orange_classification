from pathlib import Path
import os
from PIL import Image
from tqdm import tqdm
import random
import json

# ========================
# PATH CONFIGURATION
# ========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset directory (your original orange images)
DATASET_DIR = PROJECT_ROOT / "dataset"

# Processed dataset output
OUTPUT_DIR = PROJECT_ROOT / "processed_dataset"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

# Image size for resizing
IMAGE_SIZE = (224, 224)

# Train/Validation split
SPLIT_RATIO = 0.8

# Summary JSON
SUMMARY_JSON = OUTPUT_DIR / "dataset_summary.json"

# ========================
# FUNCTIONS
# ========================

def create_dir(path: Path):
    """Create directory if it doesn't exist"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def preprocess_image(img_path: Path, save_path: Path):
    """Resize and save image"""
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(save_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def split_and_preprocess():
    """Split dataset into train/val, preprocess images, and save summary"""
    create_dir(TRAIN_DIR)
    create_dir(VAL_DIR)

    class_names = [d for d in os.listdir(DATASET_DIR) if (DATASET_DIR / d).is_dir()]
    dataset_summary = {"train": {}, "val": {}}

    for class_name in class_names:
        class_dir = DATASET_DIR / class_name
        images = list(class_dir.glob("*.*"))

        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class subdirectories
        train_class_dir = TRAIN_DIR / class_name
        val_class_dir = VAL_DIR / class_name
        create_dir(train_class_dir)
        create_dir(val_class_dir)

        # Process train images
        for img_path in tqdm(train_images, desc=f"Processing train/{class_name}"):
            save_path = train_class_dir / img_path.name
            preprocess_image(img_path, save_path)

        # Process val images
        for img_path in tqdm(val_images, desc=f"Processing val/{class_name}"):
            save_path = val_class_dir / img_path.name
            preprocess_image(img_path, save_path)

        # Update summary
        dataset_summary["train"][class_name] = len(train_images)
        dataset_summary["val"][class_name] = len(val_images)

    # Save JSON summary
    with open(SUMMARY_JSON, "w") as f:
        json.dump(dataset_summary, f, indent=4)

    print(f"✅ Preprocessing complete! Images saved in: {OUTPUT_DIR}")
    print(f"📄 Dataset summary saved to: {SUMMARY_JSON}")

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print(f"Dataset root: {DATASET_DIR}")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    split_and_preprocess()
