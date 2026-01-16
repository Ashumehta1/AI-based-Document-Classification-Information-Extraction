from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.ocr import extract_text
from app.extractor import extract_fields
from app.vision_classifier import classify_image
from app.logger import init_db, log_prediction

# Initialize DB
init_db()

app = FastAPI(title="Document AI")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read image
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Document classification
        doc_type = classify_image(img)

        # OCR
        text = extract_text(img)

        # Field extraction
        fields = extract_fields(text)

        # Log prediction
        log_prediction(file.filename, doc_type, fields)

        return {
            "document_type": doc_type,
            "extracted_fields": fields
        }

    except Exception as e:
        return {"error": str(e)}
