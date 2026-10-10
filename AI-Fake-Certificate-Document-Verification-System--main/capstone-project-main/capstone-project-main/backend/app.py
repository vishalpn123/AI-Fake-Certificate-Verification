"""
app.py — Flask Backend Server
AI Fake Certificate & Document Verification System

Purpose:
    Serves the REST API consumed by the frontend.
    Accepts uploaded documents (image/PDF), runs the AI pipeline,
    and returns a JSON result.

Endpoints:
    POST /verify  — Upload document, get verification result
    GET  /health  — Health check
    GET  /stats   — Basic server statistics

Author: SUGENA MUNAVARA, VIBIN, VIJAY RAGAVENTHIRAN
College: Sri Krishna College of Technology, Coimbatore
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime

from flask       import Flask, request, jsonify
from flask_cors  import CORS
from PIL         import Image
from werkzeug.utils import secure_filename

# ── Allow imports from backend/ root ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_engine.preprocess      import preprocess
from ai_engine.ocr             import extract_ocr_features
from ai_engine.feature_extract import extract_features
from ai_engine.detect_fake     import predict, get_model


# ── Flask App Setup ───────────────────────────────────────────
app = Flask(__name__)

# Allow all origins (so HTML files opened directly can call the API)
# In production, restrict to your domain only
CORS(app, origins=['*'])

# ── Configuration ─────────────────────────────────────────────
UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_IMAGES = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
ALLOWED_PDF    = {'pdf'}
MAX_FILE_SIZE  = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-memory stats counter ────────────────────────────────────
stats = {
    'total':   0,
    'genuine': 0,
    'fake':    0,
    'history': []   # list of last 100 results
}


# ── Helper: allowed file extension ───────────────────────────
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_IMAGES | ALLOWED_PDF


# ── Helper: convert PDF to PIL Image ─────────────────────────
def pdf_to_image(pdf_path):
    """
    Convert the first page of a PDF to a PIL Image.
    Requires poppler-utils to be installed on the system.
    """
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
        if pages:
            return pages[0]
        else:
            raise ValueError("PDF has no pages")
    except ImportError:
        raise RuntimeError(
            "pdf2image is not installed or poppler is missing. "
            "Run: pip install pdf2image  and  sudo apt install poppler-utils"
        )


# ── Helper: load image from upload ───────────────────────────
def load_uploaded_image(filepath, filename):
    """
    Load image from uploaded file path.
    Handles both image files and PDFs.

    Returns: PIL Image
    """
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return pdf_to_image(filepath)
    else:
        return Image.open(filepath).convert('RGB')


# ════════════════════════════════════════════════════════════════
#  ENDPOINT: POST /verify
#  Accept a document file, run the full AI pipeline, return JSON
# ════════════════════════════════════════════════════════════════
@app.route('/verify', methods=['POST'])
def verify():
    """
    Main verification endpoint.

    Request (multipart/form-data):
        file     : document image (JPG/PNG/PDF)
        doc_type : optional string — 'certificate', 'marksheet', 'id', 'other'

    Response (JSON):
        {
          "result":     "Genuine" | "Fake",
          "confidence": 85,
          "reason":     "Document appears authentic. OCR confidence is high...",
          "ocr_text":   "CERTIFICATE OF COMPLETION ...",
          "features":   { ... }
        }
    """
    # ── Check file in request ────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send file in form-data.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use JPG, PNG, or PDF.'}), 415

    # ── Save uploaded file ───────────────────────────────────
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath    = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    doc_type = request.form.get('doc_type', 'certificate')

    try:
        # ── Step 1: Load image ───────────────────────────────
        print(f"[APP] Processing: {filename} (type={doc_type})")
        pil_img = load_uploaded_image(filepath, filename)

        # ── Step 2: Preprocess ───────────────────────────────
        preprocessed = preprocess(pil_img)
        print(f"[APP]  Preprocessed: noise={preprocessed['noise_score']:.3f}, "
              f"brightness={preprocessed['brightness']:.3f}")

        # ── Step 3: OCR ──────────────────────────────────────
        ocr_result = extract_ocr_features(preprocessed['processed_img'])
        print(f"[APP]  OCR: words={ocr_result['word_count']}, "
              f"confidence={ocr_result['ocr_confidence']:.1f}%")

        # ── Step 4: Feature extraction ───────────────────────
        feat_result = extract_features(preprocessed, ocr_result)
        print(f"[APP]  Features extracted: {len(feat_result['vector'])} dims")

        # ── Step 5: Predict ──────────────────────────────────
        prediction = predict(
            feature_vector = feat_result['vector'],
            named_features = feat_result['named'],
            ocr_issues     = ocr_result['suspicious_issues']
        )
        print(f"[APP]  Result: {prediction['result']} "
              f"({prediction['confidence']}% confidence)")

        # ── Update server stats ──────────────────────────────
        stats['total'] += 1
        if prediction['result'] == 'Genuine':
            stats['genuine'] += 1
        else:
            stats['fake'] += 1

        history_entry = {
            'filename':   filename,
            'result':     prediction['result'],
            'confidence': prediction['confidence'],
            'reason':     prediction['reason'],
            'timestamp':  datetime.now().isoformat()
        }
        stats['history'].insert(0, history_entry)
        if len(stats['history']) > 100:
            stats['history'].pop()

        # ── Build response ───────────────────────────────────
        response = {
            'result':      prediction['result'],
            'confidence':  prediction['confidence'],
            'reason':      prediction['reason'],
            'ocr_text':    ocr_result['text'][:500],  # first 500 chars
            'features': {
                'ocr_confidence':    feat_result['named']['ocr_confidence'],
                'word_count':        feat_result['named']['word_count'],
                'noise_score':       feat_result['named']['noise_score'],
                'brightness':        feat_result['named']['brightness'],
                'edge_density':      feat_result['named']['edge_density'],
                'whitespace_ratio':  feat_result['named']['whitespace_ratio'],
                'has_institution':   feat_result['named']['has_institution'],
                'reasons':           feat_result['reasons']
            },
            'metadata': {
                'filename':   filename,
                'doc_type':   doc_type,
                'timestamp':  history_entry['timestamp']
            }
        }

        return jsonify(response), 200

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[APP] ERROR: {e}\n{tb}")
        return jsonify({
            'error':   str(e),
            'details': tb
        }), 500

    finally:
        # Clean up uploaded file after processing
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /health
#  Simple health check
# ════════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    """
    Returns server health status and model load status.
    Useful for frontend to verify backend is running.
    """
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception:
        pass

    return jsonify({
        'status':       'ok',
        'model_loaded': model_loaded,
        'version':      '1.0.0',
        'timestamp':    datetime.now().isoformat()
    })


# ════════════════════════════════════════════════════════════════
#  ENDPOINT: GET /stats
#  Server-side verification statistics
# ════════════════════════════════════════════════════════════════
@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns verification statistics from this server session."""
    fake_rate = 0
    if stats['total'] > 0:
        fake_rate = round((stats['fake'] / stats['total']) * 100, 1)

    return jsonify({
        'total':     stats['total'],
        'genuine':   stats['genuine'],
        'fake':      stats['fake'],
        'fake_rate': fake_rate,
        'history':   stats['history'][:20]  # last 20 records
    })


# ════════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  AI Fake Certificate Verification System — Backend")
    print("  Sri Krishna College of Technology, Coimbatore")
    print("=" * 60)
    print()

    # Pre-load model on startup (avoids delay on first request)
    print("[APP] Pre-loading AI model…")
    try:
        get_model()
        print("[APP] ✅ Model loaded.")
    except Exception as e:
        print(f"[APP] ⚠ Model not loaded: {e}")
        print("[APP]   Run:  python ai_engine/train_model.py  to train first")

    print()
    print("[APP] Starting Flask server on http://localhost:5000")
    print("[APP] Press CTRL+C to stop")
    print()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,          # Set to False in production
        threaded=True        # Handle concurrent requests
    )
