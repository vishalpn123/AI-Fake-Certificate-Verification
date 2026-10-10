# AI Fake Certificate & Document Verification System

**Sri Krishna College of Technology, Coimbatore**  
Department of CSE | Guided by: Dr. Selina Ruby

**Team:**
- 727823TUCS358 — Vishal PN
- 727823TUCS350 — Vibin

---

## Project Overview

An end-to-end AI-powered system to detect fake and forged certificates using:
- **OCR** (Tesseract) for text extraction
- **Image preprocessing** (OpenCV + PIL) for noise removal and enhancement
- **Feature engineering** (edge density, layout analysis, brightness variance)
- **Machine Learning** (RandomForestClassifier via scikit-learn)
- **Flask** REST API backend + HTML/CSS/JS frontend

---

## Folder Structure

```
project/
├── frontend/
│   ├── login.html        ← Login page (username, password, role)
│   ├── home.html         ← Document upload page
│   ├── result.html       ← Verification result (Genuine/Fake + confidence)
│   ├── dashboard.html    ← Admin dashboard with stats and history table
│   └── style.css         ← Unified dark-theme stylesheet
│
└── backend/
    ├── app.py            ← Flask server (POST /verify, GET /health, GET /stats)
    ├── requirements.txt  ← Python dependencies
    │
    ├── ai_engine/
    │   ├── preprocess.py      ← Image preprocessing pipeline
    │   ├── ocr.py             ← Tesseract OCR integration
    │   ├── feature_extract.py ← 12-dim feature vector extraction
    │   ├── detect_fake.py     ← ML model inference + rule engine
    │   └── train_model.py     ← Model training script
    │
    └── dataset/
        ├── genuine/      ← Place 5+ real certificate images here
        └── fake/         ← Place 5+ forged certificate images here
```

---

## Prerequisites

### System-level (install once)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler:   https://github.com/oschwartz10612/poppler-windows

---

## Setup & Run Instructions

### Step 1 — Clone / download the project
```bash
cd project/backend
```

### Step 2 — Create Python virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# OR
venv\Scripts\activate             # Windows
```

### Step 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add training images (optional but recommended)
```
Copy real certificate images  → backend/dataset/genuine/
Copy forged certificate images → backend/dataset/fake/
```
If you skip this step, the system auto-generates synthetic training data for demo.

### Step 5 — Train the AI model
```bash
cd backend/
python ai_engine/train_model.py
```
This creates `backend/ai_engine/model.pkl`.

### Step 6 — Start the Flask backend
```bash
python app.py
```
Server runs at: **http://localhost:5000**

### Step 7 — Open the frontend
Open `frontend/login.html` in your browser.  
Use demo credentials: `admin / admin123`

---

## Demo Credentials

| Username  | Password   | Role        |
|-----------|------------|-------------|
| admin     | admin123   | Admin       |
| verifier  | verify123  | Verifier    |

---

## API Endpoints

| Method | URL       | Description                        |
|--------|-----------|------------------------------------|
| POST   | /verify   | Upload document, get result JSON   |
| GET    | /health   | Server health check                |
| GET    | /stats    | Session-level verification stats   |

### POST /verify — Example

**Request (multipart/form-data):**
- `file`: image file (JPG/PNG/PDF)
- `doc_type`: "certificate" | "marksheet" | "id" | "other"

**Response:**
```json
{
  "result": "Genuine",
  "confidence": 87,
  "reason": "Document appears authentic. OCR confidence is high...",
  "ocr_text": "CERTIFICATE OF COMPLETION This is to certify...",
  "features": {
    "ocr_confidence": 83.4,
    "word_count": 120,
    "noise_score": 8.2,
    "has_institution": true
  }
}
```

---

## How the AI Works

```
[Document Upload]
       │
       ▼
[Preprocessing]  ← Resize → Grayscale → Noise removal → Contrast enhancement
       │
       ▼
[OCR Module]     ← Tesseract extracts text + confidence scores
       │
       ▼
[Feature Extract] ← 12 features: OCR conf, noise, edges, brightness, etc.
       │
       ▼
[ML Model]       ← RandomForestClassifier predicts Genuine (1) or Fake (0)
       │
       ▼
[Rule Engine]    ← Heuristic corrections based on strong signals
       │
       ▼
[Result + Reason] → JSON response to frontend
```

---

## Features Extracted (12 dimensions)

| # | Feature             | Description                           |
|---|---------------------|---------------------------------------|
| 1 | OCR Confidence      | Text clarity / readability            |
| 2 | Word Count          | Document text density                 |
| 3 | Char Count          | Character richness                    |
| 4 | Noise Score         | Image noise level                     |
| 5 | Brightness          | Overall image brightness              |
| 6 | Edge Density        | Layout sharpness via Canny            |
| 7 | Horizontal Lines    | Document structure via Hough lines    |
| 8 | Texture Variance    | Background uniformity                 |
| 9 | Region Var          | Brightness consistency (tamper check) |
|10 | Whitespace Ratio    | Expected white space in documents     |
|11 | Has Institution     | Institution keyword in OCR text       |
|12 | Suspicious Issues   | OCR-detected anomaly count            |

---

## Accuracy Note

With 5 demo samples + synthetic augmentation: ~82–90% accuracy  
With 500+ real samples per class: ~95–98% accuracy  

The code includes comments explaining this. Synthetic augmentation is used  
only as a bridge for demo purposes.

---

## SDG Alignment

- **SDG 4** — Quality Education: ensures certificate authenticity
- **SDG 9** — Innovation: AI-based digital infrastructure
- **SDG 16** — Strong Institutions: reduces fraud

---

## License

Free and open-source. For academic/educational use at Sri Krishna College of Technology.
