"""
detect_fake.py — Inference / Prediction Module
AI Fake Certificate & Document Verification System

Purpose:
    Given a feature vector (extracted from a new document),
    load the trained model and predict:
      - Result: "Genuine" or "Fake"
      - Confidence: percentage (0–100)
      - Reason: human-readable explanation of the decision

This module is called by app.py for every verification request.

Author: SUGENA MUNAVARA, VIBIN, VIJAY RAGAVENTHIRAN
College: Sri Krishna College of Technology, Coimbatore
"""

import os
import joblib
import numpy as np


# ── Path to saved model ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ai_engine', 'model.pkl')

# ── Label mapping ────────────────────────────────────────────
LABEL_MAP = {1: 'Genuine', 0: 'Fake'}


def load_model():
    """
    Load the trained scikit-learn pipeline from disk.
    If model file doesn't exist, trigger training first.

    Returns: loaded sklearn Pipeline object
    """
    if not os.path.exists(MODEL_PATH):
        print("[DETECT] model.pkl not found — running training…")
        # Lazy import to avoid circular dependency
        from ai_engine.train_model import train
        train()

    try:
        model = joblib.load(MODEL_PATH)
        print("[DETECT] Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"[DETECT] Failed to load model: {e}")


# ── Singleton model cache (load once per server session) ─────
_model = None

def get_model():
    """Return cached model, loading it if not already loaded."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(feature_vector, named_features=None, ocr_issues=None):
    """
    Run the ML model on a feature vector and return a structured result.

    Args:
        feature_vector: list or array of 12 floats
        named_features: dict of feature_name → value (for reason generation)
        ocr_issues:     list of OCR-detected anomaly strings

    Returns:
        dict:
          'result'     — "Genuine" or "Fake"
          'confidence' — integer 0–100
          'reason'     — human-readable string explaining the decision
    """
    model = get_model()

    X = np.array([feature_vector], dtype=np.float32)

    # ── Prediction ───────────────────────────────────────────
    pred_label   = model.predict(X)[0]              # 0 or 1
    pred_proba   = model.predict_proba(X)[0]         # [P(fake), P(genuine)]

    result       = LABEL_MAP[int(pred_label)]
    confidence   = round(float(max(pred_proba)) * 100, 1)

    # ── Post-process: apply rule-based confidence corrections ─
    # Rule engine can override ML when we have strong heuristic signals
    confidence, result = apply_rules(feature_vector, result, confidence, named_features, ocr_issues)

    # ── Generate human-readable reason ───────────────────────
    reason = build_reason(result, feature_vector, named_features, ocr_issues)

    return {
        'result':     result,
        'confidence': int(round(confidence)),
        'reason':     reason
    }


def apply_rules(feature_vector, result, confidence, named, ocr_issues):
    """
    Rule-based post-processing layer on top of ML prediction.
    Adjusts confidence up/down based on strong signal features.

    Feature index reference (must match feature_extract.py):
        0: OCR confidence
        1: Word count (normalized)
        3: Noise score
        8: Region brightness variance
        10: Has institution keyword
        11: Suspicious issues count
    """
    if named is None:
        return confidence, result

    ocr_conf     = feature_vector[0]
    noise        = feature_vector[3]
    region_var   = feature_vector[8]
    has_inst     = feature_vector[10]
    issue_frac   = feature_vector[11]

    # Rule 1: Low OCR confidence → treat as fake
    # raised threshold to catch more blurry/edited documents
    if ocr_conf < 0.55 and result == 'Genuine':
        result     = 'Fake'
        confidence = max(confidence, 70.0)

    # Rule 2: High noise → likely tampered
    if noise > 0.50 and result == 'Genuine':
        result     = 'Fake'
        confidence = max(confidence, 72.0)

    # Rule 3: Strong regional variance → tamper evidence
    if region_var > 0.40 and result == 'Genuine':
        confidence = min(confidence - 10, 60.0)

    # Rule 4: Multiple OCR issues → fake confidence boost
    if issue_frac > 0.2 and result == 'Genuine':
        confidence = max(confidence - 8, 55.0)

    # Rule 5: Clean document (high OCR + low noise + has institution) → genuine boost
    if ocr_conf > 0.80 and noise < 0.20 and has_inst == 1.0:
        if result == 'Genuine':
            confidence = min(confidence + 5, 99.0)

    # Rule 6: Basic heuristic override – if any strong negative signal is present
    # (no institution, low OCR, high noise, or OCR issues) then mark fake
    if result == 'Genuine' and (
        has_inst < 0.5 or
        ocr_conf < 0.60 or
        noise > 0.55 or
        issue_frac > 0.15
    ):
        result     = 'Fake'
        confidence = max(confidence, 65.0)

    # Clamp confidence to valid range
    confidence = float(max(50.0, min(99.5, confidence)))

    # Log final decision if it was overridden by rules (useful during demo/debug)
    if hasattr(model, 'feature_importances_'):
        # print summary only when real model present
        if result == 'Fake' and pred_label == 1:
            print(f"[DETECT] Rule override: genuine→fake (ocr_conf={ocr_conf:.2f}, noise={noise:.2f}, has_inst={has_inst}, issues={issue_frac:.2f})")
        elif result == 'Genuine' and pred_label == 0:
            print(f"[DETECT] Rule override: fake→genuine (ocr_conf={ocr_conf:.2f}, noise={noise:.2f}, has_inst={has_inst})")

    return confidence, result


def build_reason(result, feature_vector, named, ocr_issues):
    """
    Build a natural language explanation for the prediction.

    Args:
        result:         "Genuine" or "Fake"
        feature_vector: list of 12 floats
        named:          dict of named feature values
        ocr_issues:     list of OCR anomaly strings

    Returns:
        str: explanation paragraph
    """
    ocr_conf   = feature_vector[0]
    noise      = feature_vector[3]
    region_var = feature_vector[8]
    whitespace = feature_vector[9]
    has_inst   = feature_vector[10]

    parts = []

    if result == 'Genuine':
        parts.append("Document appears authentic.")

        if ocr_conf > 0.75:
            parts.append("OCR text extraction yielded high confidence — text is clearly readable and consistent.")
        if noise < 0.25:
            parts.append("Image noise levels are within acceptable range for a printed document.")
        if region_var < 0.25:
            parts.append("Regional brightness distribution is uniform — no signs of copy-paste editing.")
        if has_inst == 1.0:
            parts.append("Institution/organization name was detected in the document content.")
        if whitespace > 0.5:
            parts.append("Document layout includes appropriate white space consistent with standard templates.")

    else:  # Fake
        parts.append("Document shows signs of tampering or forgery.")

        if ocr_conf < 0.55:
            parts.append("OCR confidence is low — text may have been digitally altered or is inconsistent.")
        if noise > 0.40:
            parts.append("Elevated image noise detected — possible JPEG compression artifacts from re-editing.")
        if region_var > 0.35:
            parts.append("Uneven regional brightness detected — this may indicate image composition or copy-paste manipulation.")
        if has_inst < 0.5:
            parts.append("No institution or issuing organization name found in extracted text.")

        if ocr_issues:
            parts.append(f"OCR anomalies detected: {'; '.join(ocr_issues[:3])}.")

    if not parts:
        parts.append(f"Classification: {result}. Analysis based on OCR confidence, image noise, and layout features.")

    return ' '.join(parts)
