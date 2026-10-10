"""
ocr.py — OCR (Optical Character Recognition) Module
AI Fake Certificate & Document Verification System

Purpose:
    Extract text from preprocessed images using Tesseract OCR.
    Also compute OCR confidence score and detect text anomalies.

Key outputs:
    - Extracted text string
    - OCR confidence score (0–100)
    - Word count, line count, character count
    - Suspicious text patterns (anomaly flags)

Author: SUGENA MUNAVARA, VIBIN, VIJAY RAGAVENTHIRAN
College: Sri Krishna College of Technology, Coimbatore
"""

import re
import pytesseract
from PIL import Image


# ── Tesseract Config ─────────────────────────────────────────────────────────
# Page Segmentation Modes (PSM):
#   3 = Fully automatic page segmentation (default) — best for full certificates
#   6 = Assume a single uniform block of text
# OEM 3 = Default engine mode (LSTM + legacy)

TESSERACT_CONFIG = '--psm 3 --oem 3'


def run_ocr(pil_img):
    """
    Run Tesseract OCR on a PIL Image and return raw text.

    Args:
        pil_img: PIL Image (processed grayscale recommended)

    Returns:
        str: extracted text (may be empty if OCR finds nothing)
    """
    try:
        text = pytesseract.image_to_string(pil_img, config=TESSERACT_CONFIG)
        return text.strip()
    except Exception as e:
        print(f"[OCR] Tesseract error: {e}")
        return ""


def run_ocr_with_confidence(pil_img):
    """
    Run OCR and also get per-word confidence scores.
    Tesseract returns a data frame (as string) with bounding boxes and confidence.

    Returns:
        dict with:
          'text'       — full extracted text string
          'confidence' — average confidence score (0–100); higher = more readable
          'words'      — list of individual words detected
    """
    try:
        # image_to_data returns detailed info per word
        data = pytesseract.image_to_data(
            pil_img,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )

        # Extract valid words and their confidence scores
        valid_words  = []
        confidences  = []

        for i, conf in enumerate(data['conf']):
            try:
                conf_val = int(conf)
            except (ValueError, TypeError):
                continue

            if conf_val > 0:  # -1 means no text
                word = data['text'][i].strip()
                if word:
                    valid_words.append(word)
                    confidences.append(conf_val)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        full_text      = ' '.join(valid_words)

        return {
            'text':       full_text,
            'confidence': round(avg_confidence, 2),
            'words':      valid_words
        }

    except Exception as e:
        print(f"[OCR] Confidence extraction error: {e}")
        # Fallback to simple OCR
        text = run_ocr(pil_img)
        return {
            'text':       text,
            'confidence': 50.0,  # neutral fallback
            'words':      text.split()
        }


def count_text_stats(text):
    """
    Compute basic statistics about the extracted text.
    These stats help identify if a document has expected content density.

    Args:
        text: str — OCR extracted text

    Returns:
        dict with character_count, word_count, line_count, paragraph_count
    """
    lines      = [l for l in text.splitlines() if l.strip()]
    words      = text.split()
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    return {
        'character_count': len(text),
        'word_count':      len(words),
        'line_count':      len(lines),
        'paragraph_count': len(paragraphs)
    }


def detect_suspicious_patterns(text):
    """
    Scan OCR text for patterns that are common in fake or tampered certificates.
    Returns a list of detected issues (empty list = no issues found).

    Checks:
        - Very low word count (blank/empty document)
        - No institution keywords
        - Repeated characters (possible pixelation artifact)
        - Suspicious number formatting
    """
    issues = []
    text_lower = text.lower()

    # Check 1: Almost empty document
    if len(text.split()) < 5:
        issues.append("Very low text content detected — document may be blank or unreadable")

    # Check 2: Expected certificate keywords missing
    certificate_keywords = [
        'certificate', 'degree', 'awarded', 'completed', 'university',
        'college', 'institute', 'school', 'board', 'examination',
        'result', 'mark', 'grade', 'pass', 'student'
    ]
    found_keywords = [kw for kw in certificate_keywords if kw in text_lower]
    if not found_keywords and len(text.split()) > 10:
        issues.append("No standard certificate/document keywords found")

    # Check 3: Suspicious repetition (common in edited/pasted text)
    # Look for sequences of repeated unusual characters
    if re.search(r'(.)\1{5,}', text):
        issues.append("Repeated character sequences detected — possible image artifact")

    # Check 4: Unusual date patterns (very far future or past)
    years = re.findall(r'\b(19[0-9]{2}|20[0-9]{2})\b', text)
    if years:
        year_ints = [int(y) for y in years]
        if any(y > 2030 or y < 1950 for y in year_ints):
            issues.append("Unusual year detected in document")

    return issues


def extract_ocr_features(pil_img):
    """
    Master OCR function — runs all checks and returns a comprehensive result.

    Args:
        pil_img: PIL Image (preprocessed)

    Returns:
        dict:
          'text'               — full extracted text
          'ocr_confidence'     — 0–100 (higher = text is clearer/more readable)
          'word_count'         — total words
          'line_count'         — total lines
          'suspicious_issues'  — list of anomaly strings
          'has_institution'    — bool: found institution name?
          'normalized_conf'    — 0.0–1.0 normalized confidence
    """
    # Run OCR with confidence
    ocr_result = run_ocr_with_confidence(pil_img)
    text       = ocr_result['text']
    confidence = ocr_result['confidence']

    # Text statistics
    stats = count_text_stats(text)

    # Suspicious pattern check
    issues = detect_suspicious_patterns(text)

    # Check for institution keyword
    institution_keywords = ['university', 'college', 'institute', 'board', 'school', 'academy']
    has_institution = any(kw in text.lower() for kw in institution_keywords)

    return {
        'text':              text,
        'ocr_confidence':    confidence,          # 0–100
        'word_count':        stats['word_count'],
        'line_count':        stats['line_count'],
        'character_count':   stats['character_count'],
        'suspicious_issues': issues,
        'has_institution':   has_institution,
        'normalized_conf':   confidence / 100.0   # normalized to 0–1
    }
