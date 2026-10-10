"""
feature_extract.py — Feature Extraction Module
AI Fake Certificate & Document Verification System

Purpose:
    Extract a numerical feature vector from a document image.
    These features are fed into the ML model for classification.

Features extracted:
    1.  OCR confidence (text clarity)
    2.  Word count (document density)
    3.  Character count
    4.  Noise level (from preprocessing)
    5.  Brightness level
    6.  Edge density (layout sharpness)
    7.  Horizontal line count (layout structure)
    8.  Texture variance (uniformity)
    9.  Region brightness variance (tamper indicator)
    10. White space ratio
    11. Has institution keyword (1 or 0)
    12. Suspicious OCR issues count

Author: SUGENA MUNAVARA, VIBIN, VIJAY RAGAVENTHIRAN
College: Sri Krishna College of Technology, Coimbatore
"""

import cv2
import numpy as np
from PIL import Image


# ── Feature Vector length ────────────────────────────────────
FEATURE_DIM = 12   # Total number of features in our vector


def compute_edge_density(pil_img):
    """
    Detect edges using Canny edge detection.
    Genuine documents have consistent, clear edges.
    Tampered documents may have extra or irregular edges.

    Returns: float [0, 1] — ratio of edge pixels to total pixels
    """
    np_img = np.array(pil_img.convert('L'), dtype=np.uint8)
    edges  = cv2.Canny(np_img, threshold1=50, threshold2=150)
    edge_ratio = float(np.sum(edges > 0)) / (np_img.shape[0] * np_img.shape[1])
    return edge_ratio


def count_horizontal_lines(pil_img):
    """
    Detect horizontal lines in the document using Hough Line Transform.
    Genuine certificates typically have a consistent number of horizontal
    dividers (lines under headings, borders, etc.).

    Returns: int — number of detected horizontal lines
    """
    np_img = np.array(pil_img.convert('L'), dtype=np.uint8)

    # Canny edges
    edges = cv2.Canny(np_img, 50, 150, apertureSize=3)

    # Hough lines — only near-horizontal lines (angle ±10°)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=20
    )

    if lines is None:
        return 0

    horizontal = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:  # near horizontal
            horizontal += 1

    return horizontal


def compute_texture_variance(pil_img):
    """
    Compute variance of pixel intensities in local regions.
    Low variance = uniform background (typical of genuine printed docs).
    High variance = irregular texture (may indicate tampering or editing).

    Returns: float — normalized variance [0, 1]
    """
    np_img  = np.array(pil_img.convert('L'), dtype=np.float32)
    variance = float(np.var(np_img))
    # Normalize: typical document variance ~ 500–3000; cap at 5000
    return min(variance / 5000.0, 1.0)


def compute_region_brightness_variance(pil_img, grid=4):
    """
    Divide image into a grid and compute brightness in each cell.
    Then compute variance of cell brightnesses.

    Genuine docs: brightness is fairly uniform (slight borders darker).
    Tampered docs: localized bright/dark regions from copy-paste.

    Returns: float [0, 1] — higher = more uneven brightness
    """
    np_img = np.array(pil_img.convert('L'), dtype=np.float32)
    h, w   = np_img.shape
    cell_h = h // grid
    cell_w = w // grid

    cell_means = []
    for r in range(grid):
        for c in range(grid):
            cell = np_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            if cell.size > 0:
                cell_means.append(float(np.mean(cell)))

    if not cell_means:
        return 0.0

    variance = float(np.var(cell_means))
    return min(variance / 2000.0, 1.0)


def compute_whitespace_ratio(pil_img, threshold=240):
    """
    Compute fraction of near-white pixels.
    Genuine certificates usually have significant white space.
    Heavily edited/fake docs might have unusual white space patterns.

    Returns: float [0, 1]
    """
    np_img   = np.array(pil_img.convert('L'), dtype=np.uint8)
    white    = np.sum(np_img >= threshold)
    total    = np_img.size
    return float(white) / total if total > 0 else 0.5


def extract_features(preprocessed_dict, ocr_features_dict):
    """
    Build the complete feature vector by combining:
        - preprocessed image metrics (noise, brightness, etc.)
        - OCR results (confidence, word count, etc.)
        - derived image features (edges, lines, texture)

    Args:
        preprocessed_dict: output from preprocess.preprocess()
        ocr_features_dict: output from ocr.extract_ocr_features()

    Returns:
        dict:
          'vector'     — list of 12 floats (the feature vector for ML)
          'named'      — dict of feature_name → value (for explanations)
          'reasons'    — list of human-readable anomaly strings
    """
    pil_img    = preprocessed_dict['processed_img']
    noise      = preprocessed_dict['noise_score']
    brightness = preprocessed_dict['brightness']

    ocr_conf   = ocr_features_dict['normalized_conf']         # 0–1
    word_count = min(ocr_features_dict['word_count'] / 300.0, 1.0)  # normalize by 300 words
    char_count = min(ocr_features_dict['character_count'] / 1500.0, 1.0)
    has_inst   = float(ocr_features_dict['has_institution'])
    issue_cnt  = min(len(ocr_features_dict['suspicious_issues']) / 5.0, 1.0)

    # Image-level features
    edge_density    = compute_edge_density(pil_img)
    h_lines         = min(count_horizontal_lines(pil_img) / 20.0, 1.0)  # norm
    texture_var     = compute_texture_variance(pil_img)
    region_var      = compute_region_brightness_variance(pil_img)
    whitespace      = compute_whitespace_ratio(pil_img)

    # ── Feature Vector (order matters — must match training) ─
    vector = [
        ocr_conf,       # F1: OCR text clarity
        word_count,     # F2: Document density
        char_count,     # F3: Character richness
        noise,          # F4: Image noise level
        brightness,     # F5: Overall brightness
        edge_density,   # F6: Edge sharpness
        h_lines,        # F7: Horizontal structure
        texture_var,    # F8: Texture uniformity
        region_var,     # F9: Regional brightness variance
        whitespace,     # F10: White space ratio
        has_inst,       # F11: Institution keyword present
        issue_cnt       # F12: Number of suspicious OCR issues
    ]

    # ── Named features for explainability ───────────────────
    named = {
        'ocr_confidence':       round(ocr_conf * 100, 1),
        'word_count':           ocr_features_dict['word_count'],
        'character_count':      ocr_features_dict['character_count'],
        'noise_score':          round(noise * 100, 1),
        'brightness':           round(brightness * 100, 1),
        'edge_density':         round(edge_density * 100, 1),
        'horizontal_lines':     count_horizontal_lines(pil_img),
        'texture_variance':     round(texture_var * 100, 1),
        'region_brightness_var':round(region_var * 100, 1),
        'whitespace_ratio':     round(whitespace * 100, 1),
        'has_institution':      bool(has_inst),
        'suspicious_issues_cnt':ocr_features_dict['suspicious_issues']
    }

    # ── Generate human-readable reasons ─────────────────────
    reasons = []

    if ocr_conf < 0.5:
        reasons.append("Low OCR confidence — text may be blurry or edited")
    if noise > 0.5:
        reasons.append("High image noise — possible compression artifacts or tampering")
    if region_var > 0.4:
        reasons.append("Uneven regional brightness — may indicate copy-paste editing")
    if not bool(has_inst):
        reasons.append("No institution name detected via OCR")
    if issue_cnt > 0.2:
        reasons.append(f"Suspicious text patterns: {ocr_features_dict['suspicious_issues']}")
    if texture_var > 0.6:
        reasons.append("High texture variance — inconsistent background detected")

    if not reasons:
        reasons.append("No significant anomalies detected")

    return {
        'vector':  vector,
        'named':   named,
        'reasons': reasons
    }
