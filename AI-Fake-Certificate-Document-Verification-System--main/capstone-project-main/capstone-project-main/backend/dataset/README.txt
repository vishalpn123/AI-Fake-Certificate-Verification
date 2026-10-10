DATASET INSTRUCTIONS
====================

This folder holds the training images for the AI model.

FOLDER STRUCTURE:
    dataset/
    ├── genuine/    ← Place REAL certificate images here (min 5)
    └── fake/       ← Place TAMPERED/FAKE certificate images here (min 5)

SUPPORTED FILE FORMATS:
    .jpg, .jpeg, .png, .bmp, .tiff

MINIMUM RECOMMENDED:
    5 images per class (for demo)
    500+ images per class (for production ~98% accuracy)

HOW TO ADD IMAGES:
    1. Scan or photograph real certificates → put in genuine/
    2. Edit copies of certificates (change marks, name, etc.) → put in fake/
    3. Run training: cd backend/ && python ai_engine/train_model.py

NOTE:
    If this folder is empty, the system auto-generates a synthetic
    training dataset for demo purposes.
    Real images are STRONGLY recommended for production use.
