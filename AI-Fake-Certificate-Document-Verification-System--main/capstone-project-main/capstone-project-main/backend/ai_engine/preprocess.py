"""
preprocess.py — Image Preprocessing Module
AI Fake Certificate & Document Verification System

Purpose:
    Accepts a raw image (PIL Image or filepath) and returns
    a cleaned, enhanced image ready for OCR and feature extraction.

Steps:
    1. Resize to a standard width (preserving aspect ratio)
    2. Convert to grayscale
    3. Remove noise (Gaussian blur + adaptive threshold)
    4. Enhance contrast
    5. Return processed image (PIL format) + metadata dict

Author: SUGENA MUNAVARA, VIBIN, VIJAY RAGAVENTHIRAN
College: Sri Krishna College of Technology, Coimbatore
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ── Constants ────────────────────────────────────────────────────────────────
TARGET_WIDTH = 1200      # Standard width for processing
NOISE_KSIZE  = (3, 3)   # Gaussian blur kernel size


def load_image(image_input):
    """
    Load image from:
      - PIL Image object  → returned as-is
      - file path string  → opened with PIL

    Returns: PIL Image (RGB mode)
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert('RGB')
    elif isinstance(image_input, str):
        return Image.open(image_input).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")


def resize_to_standard(pil_img, target_width=TARGET_WIDTH):
    """
    Resize image to a fixed width while preserving aspect ratio.
    This ensures all images have consistent resolution for analysis.

    Returns: resized PIL Image
    """
    original_w, original_h = pil_img.size
    if original_w == 0:
        return pil_img

    scale_factor = target_width / original_w
    new_h        = int(original_h * scale_factor)
    resized      = pil_img.resize((target_width, new_h), Image.LANCZOS)
    return resized


def to_grayscale(pil_img):
    """
    Convert RGB image to grayscale.
    Grayscale simplifies processing and reduces noise.

    Returns: PIL Image in 'L' (grayscale) mode
    """
    return pil_img.convert('L')


def remove_noise(pil_img):
    """
    Apply Gaussian blur to reduce high-frequency noise.
    Helps OCR and feature extraction by removing scanner/camera artifacts.

    Returns: denoised PIL Image
    """
    # Convert PIL → NumPy for OpenCV processing
    np_img = np.array(pil_img, dtype=np.uint8)

    # Gaussian blur for noise removal
    blurred = cv2.GaussianBlur(np_img, NOISE_KSIZE, 0)

    # Convert back to PIL
    return Image.fromarray(blurred)


def enhance_contrast(pil_img, factor=1.4):
    """
    Enhance image contrast to make text more readable for OCR.
    factor > 1 increases contrast; factor = 1 = no change.

    Returns: contrast-enhanced PIL Image
    """
    enhancer = ImageEnhance.Contrast(pil_img)
    return enhancer.enhance(factor)


def enhance_sharpness(pil_img, factor=1.3):
    """
    Sharpen image slightly to improve edge detection.

    Returns: sharpened PIL Image
    """
    enhancer = ImageEnhance.Sharpness(pil_img)
    return enhancer.enhance(factor)


def binarize(pil_img):
    """
    Apply adaptive thresholding to convert grayscale to binary (black/white).
    Adaptive threshold handles uneven lighting (common in scanned docs).

    Returns: binarized PIL Image
    """
    np_img = np.array(pil_img, dtype=np.uint8)

    # Adaptive threshold: better than global threshold for documents
    binary = cv2.adaptiveThreshold(
        np_img,
        255,                              # max pixel value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Gaussian-weighted neighborhood
        cv2.THRESH_BINARY,                # output type
        blockSize=31,                     # neighborhood size (odd number)
        C=10                              # constant subtracted from mean
    )
    return Image.fromarray(binary)


def measure_noise(pil_img):
    """
    Measure the amount of noise/artifacts in the image.
    High noise may indicate tampering or poor quality scanning.

    Returns: float between 0 (no noise) and 1 (very noisy)
    """
    np_img = np.array(pil_img.convert('L'), dtype=np.float32)

    # Laplacian variance — high variance = noisy, low = smooth (possibly fake)
    laplacian = cv2.Laplacian(np_img.astype(np.uint8), cv2.CV_64F)
    variance  = np.var(laplacian)

    # Normalize variance to [0, 1] range
    # Typical doc variance: ~100–2000; >3000 = very noisy
    noise_score = min(variance / 3000.0, 1.0)
    return float(noise_score)


def measure_brightness(pil_img):
    """
    Compute average brightness of the image.
    Very low or very high brightness can indicate manipulation.

    Returns: float [0, 1] — 0 = black, 1 = white
    """
    np_img = np.array(pil_img.convert('L'), dtype=np.float32)
    return float(np.mean(np_img) / 255.0)


def preprocess(image_input):
    """
    Main preprocessing pipeline. Runs all steps in order.

    Args:
        image_input: PIL Image object or filepath string

    Returns:
        dict with keys:
          'processed_img'  — final cleaned PIL Image (for OCR)
          'binary_img'     — binarized version
          'original_img'   — original PIL Image
          'noise_score'    — 0–1 noise level
          'brightness'     — 0–1 brightness level
          'dimensions'     — (width, height) after resize
    """
    # Step 1: Load
    original = load_image(image_input)

    # Step 2: Resize to standard width
    resized = resize_to_standard(original)

    # Step 3: Grayscale
    gray = to_grayscale(resized)

    # Step 4: Noise removal
    denoised = remove_noise(gray)

    # Step 5: Contrast enhancement
    enhanced = enhance_contrast(denoised, factor=1.4)

    # Step 6: Sharpness
    sharpened = enhance_sharpness(enhanced, factor=1.2)

    # Step 7: Binarize (for OCR)
    binary = binarize(sharpened)

    # Step 8: Measure image quality metrics
    noise_score = measure_noise(resized)
    brightness  = measure_brightness(resized)

    return {
        'processed_img': sharpened,     # enhanced grayscale → best for OCR
        'binary_img':    binary,         # binarized → alternative OCR input
        'original_img':  original,       # keep original for reference
        'noise_score':   noise_score,    # 0 = clean, 1 = very noisy
        'brightness':    brightness,     # 0 = dark, 1 = bright
        'dimensions':    resized.size    # (width, height)
    }
