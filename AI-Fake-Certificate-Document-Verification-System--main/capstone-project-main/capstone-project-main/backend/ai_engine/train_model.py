"""
train_model.py — Model Training Script
AI Fake Certificate & Document Verification System

Purpose:
    Train a RandomForestClassifier on genuine and fake certificate images.
    The trained model is saved to disk (model.pkl) for use during inference.

Dataset structure expected:
    backend/dataset/genuine/  ← 5+ real certificate images
    backend/dataset/fake/     ← 5+ tampered/fake certificate images

NOTE:
    5 samples per class is minimal — only for demo purposes.
    For production accuracy (~98%), use 500+ samples per class.
    The model below includes engineered features that partially compensate
    for small dataset size.

Usage:
    cd backend/
    python ai_engine/train_model.py

Author: SUGENA MUNAVARA, VIBIN, VIJAY RAGAVENTHIRAN
College: Sri Krishna College of Technology, Coimbatore
"""

import os
import sys
import joblib
import numpy as np

# Allow relative imports from backend/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.preprocess       import preprocess
from ai_engine.ocr              import extract_ocr_features
from ai_engine.feature_extract  import extract_features
from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import cross_val_score, StratifiedKFold
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import classification_report


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENUINE_DIR  = os.path.join(BASE_DIR, 'dataset', 'genuine')
FAKE_DIR     = os.path.join(BASE_DIR, 'dataset', 'fake')
MODEL_PATH   = os.path.join(BASE_DIR, 'ai_engine', 'model.pkl')

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Labels
LABEL_GENUINE = 1  # positive class
LABEL_FAKE    = 0  # negative class


def load_images_from_dir(directory, label):
    """
    Load all supported images from a directory.
    Each image is preprocessed → OCR → features extracted.

    Args:
        directory: path to image folder
        label:     int (1=genuine, 0=fake)

    Returns:
        X: list of feature vectors (each is a list of 12 floats)
        y: list of labels
    """
    X, y = [], []
    if not os.path.isdir(directory):
        print(f"[TRAIN] WARNING: Directory not found: {directory}")
        return X, y

    files = [f for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_EXT)]
    if not files:
        print(f"[TRAIN] WARNING: No images found in {directory}")
        return X, y

    for fname in files:
        fpath = os.path.join(directory, fname)
        print(f"[TRAIN] Processing: {fname} (label={'genuine' if label else 'fake'})")
        try:
            preprocessed = preprocess(fpath)
            ocr_feats     = extract_ocr_features(preprocessed['processed_img'])
            feat_result   = extract_features(preprocessed, ocr_feats)

            X.append(feat_result['vector'])
            y.append(label)
            print(f"         ✓ Feature vector: {[round(v, 3) for v in feat_result['vector']]}")
        except Exception as e:
            print(f"[TRAIN] ERROR processing {fname}: {e}")

    return X, y


def generate_synthetic_samples(base_vector, label, n=5, noise_std=0.05):
    """
    Data augmentation: generate synthetic training samples by adding
    small Gaussian noise to an existing feature vector.

    This technique helps when the dataset is very small (demo purpose).
    NOTE: In production, use real diverse samples instead.

    Args:
        base_vector: list of floats
        label:       int
        n:           number of synthetic samples to generate
        noise_std:   standard deviation of noise

    Returns:
        X_syn: list of augmented vectors
        y_syn: list of labels
    """
    X_syn, y_syn = [], []
    base = np.array(base_vector, dtype=np.float32)

    for _ in range(n):
        noise  = np.random.normal(0, noise_std, size=base.shape).astype(np.float32)
        sample = np.clip(base + noise, 0.0, 1.0).tolist()
        X_syn.append(sample)
        y_syn.append(label)

    return X_syn, y_syn


def build_fallback_dataset():
    """
    When no real images are available, generate a synthetic dataset
    based on typical feature distributions for genuine vs fake docs.

    Genuine features (typical):
        - High OCR confidence (0.85+)
        - High word count
        - Low noise
        - Consistent edges, layout

    Fake features (typical):
        - Lower OCR confidence
        - Irregular noise
        - Inconsistent brightness regions

    Returns:
        X, y as lists
    """
    print("[TRAIN] Generating synthetic training data (no images found)…")

    np.random.seed(42)
    X, y = [], []
    n_per_class = 30  # generate 30 synthetic samples per class

    for _ in range(n_per_class):
        # Genuine: clean, consistent document
        genuine_vec = [
            np.random.uniform(0.80, 0.98),   # F1: OCR confidence (high)
            np.random.uniform(0.40, 0.90),   # F2: word count (moderate-high)
            np.random.uniform(0.40, 0.90),   # F3: char count
            np.random.uniform(0.05, 0.25),   # F4: noise (low)
            np.random.uniform(0.60, 0.90),   # F5: brightness (normal)
            np.random.uniform(0.10, 0.30),   # F6: edge density
            np.random.uniform(0.20, 0.60),   # F7: horizontal lines
            np.random.uniform(0.10, 0.35),   # F8: texture variance (low-med)
            np.random.uniform(0.05, 0.20),   # F9: region brightness var (low)
            np.random.uniform(0.55, 0.80),   # F10: whitespace ratio
            1.0,                              # F11: has institution (yes)
            np.random.uniform(0.00, 0.10),   # F12: suspicious issues (none/few)
        ]
        X.append(genuine_vec)
        y.append(LABEL_GENUINE)

    for _ in range(n_per_class):
        # Fake: noisy, inconsistent document
        fake_vec = [
            np.random.uniform(0.20, 0.65),   # F1: OCR confidence (lower)
            np.random.uniform(0.10, 0.50),   # F2: word count (may be low or high)
            np.random.uniform(0.10, 0.50),   # F3: char count
            np.random.uniform(0.35, 0.80),   # F4: noise (higher)
            np.random.uniform(0.30, 0.70),   # F5: brightness (erratic)
            np.random.uniform(0.30, 0.70),   # F6: edge density (irregular)
            np.random.uniform(0.05, 0.40),   # F7: horizontal lines (inconsistent)
            np.random.uniform(0.40, 0.90),   # F8: texture variance (high)
            np.random.uniform(0.30, 0.80),   # F9: region variance (high — tampered)
            np.random.uniform(0.20, 0.55),   # F10: whitespace ratio (different)
            np.random.uniform(0.0, 0.6),     # F11: may or may not have institution
            np.random.uniform(0.20, 0.80),   # F12: more suspicious issues
        ]
        X.append(fake_vec)
        y.append(LABEL_FAKE)

    return X, y


def train():
    """
    Main training function.

    1. Load genuine and fake images from dataset/
    2. Extract features from each image
    3. Augment if dataset is small
    4. Train RandomForestClassifier
    5. Evaluate with cross-validation
    6. Save model.pkl
    """
    print("=" * 60)
    print("  AI CERTIFICATE VERIFICATION — MODEL TRAINING")
    print("=" * 60)

    # ── Load real images ─────────────────────────────────────
    X_gen, y_gen = load_images_from_dir(GENUINE_DIR, LABEL_GENUINE)
    X_fak, y_fak = load_images_from_dir(FAKE_DIR,    LABEL_FAKE)

    X_all = X_gen + X_fak
    y_all = y_gen + y_fak

    total_real = len(X_all)
    print(f"\n[TRAIN] Real samples loaded: {total_real} "
          f"(genuine={len(X_gen)}, fake={len(X_fak)})")

    # ── Augment if small dataset ─────────────────────────────
    if len(X_gen) > 0:
        for vec in X_gen[:]:
            X_aug, y_aug = generate_synthetic_samples(vec, LABEL_GENUINE, n=8)
            X_all.extend(X_aug); y_all.extend(y_aug)
    if len(X_fak) > 0:
        for vec in X_fak[:]:
            X_aug, y_aug = generate_synthetic_samples(vec, LABEL_FAKE, n=8)
            X_all.extend(X_aug); y_all.extend(y_aug)

    # ── Fallback: no images at all ───────────────────────────
    if total_real == 0:
        X_syn, y_syn = build_fallback_dataset()
        X_all.extend(X_syn); y_all.extend(y_syn)
        print("[TRAIN] Using fully synthetic data — add real images to dataset/ for better accuracy!")

    print(f"[TRAIN] Total training samples (after augmentation): {len(X_all)}")

    X_np = np.array(X_all, dtype=np.float32)
    y_np = np.array(y_all, dtype=int)

    # ── Build pipeline: Scaler + RandomForest ────────────────
    # StandardScaler ensures all features have equal scale
    # RandomForest is robust to small datasets and provides feature importance
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,    # 200 trees — more = more stable
            max_depth=8,         # limit depth to prevent overfitting
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',  # handle class imbalance
            random_state=42
        ))
    ])

    # ── Cross-validation evaluation ──────────────────────────
    if len(X_np) >= 6:
        cv = StratifiedKFold(n_splits=min(5, len(X_np) // 2), shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_np, y_np, cv=cv, scoring='accuracy')
        print(f"\n[TRAIN] Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"         Individual fold scores: {[round(s, 3) for s in cv_scores]}")

        if cv_scores.mean() < 0.80:
            print("[TRAIN] WARNING: Low accuracy — add more diverse real samples for production use!")

    # ── Final training on all data ───────────────────────────
    pipeline.fit(X_np, y_np)
    print("\n[TRAIN] Model trained on full dataset.")

    # ── Feature importance (RandomForest) ────────────────────
    feature_names = [
        'OCR Confidence', 'Word Count', 'Char Count', 'Noise', 'Brightness',
        'Edge Density', 'Horiz Lines', 'Texture Var', 'Region Var',
        'Whitespace', 'Has Institution', 'Suspicious Issues'
    ]
    importances = pipeline.named_steps['clf'].feature_importances_
    print("\n[TRAIN] Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"    {name:<25} {bar} {imp:.3f}")

    # ── Save model ───────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[TRAIN] ✅ Model saved to: {MODEL_PATH}")
    print("=" * 60)
    print("  Training complete. Run app.py to start the server.")
    print("=" * 60)


if __name__ == '__main__':
    train()
