from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms
import easyocr
import pytesseract
import re
import torch.nn as nn

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="Document Classification & Extraction")

# ----------------------------
# Simple CNN Definition
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        IMG_SIZE = 224
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# Load Vision Model (CNN)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # invoice, id_card, certificate, resume
IMG_SIZE = 224

model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(r"D:\nrt_python_assignment\models\vision\doc_classifier.pth", map_location=device))
model.eval()

classes = ["invoice", "id_card", "certificate", "resume"]

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ----------------------------
# OCR Reader
# ----------------------------
reader = easyocr.Reader(['en'])

# ----------------------------
# Helper: Extract Fields
# ----------------------------
def extract_fields(text):
    fields = {}

    # Name (simple regex for capitalized words)
    name_match = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
    fields['name'] = name_match[0] if name_match else ""

    # Date (formats like 2026-01-14, 14/01/2026)
    date_match = re.findall(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text)
    fields['date'] = date_match[0] if date_match else ""

    # Amount (₹2,500, $2500, 2500)
    amount_match = re.findall(r"₹?\$?\d[\d,]*", text)
    fields['amount'] = amount_match[0] if amount_match else ""

    # ID Number (simple pattern)
    id_match = re.findall(r"[A-Z0-9]{5,15}", text)
    fields['id_number'] = id_match[0] if id_match else ""

    return fields

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify-and-extract")
async def classify_and_extract(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Vision classification
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    document_type = classes[pred.item()]

    # OCR
    text = pytesseract.image_to_string(img)
    # Or use EasyOCR: 
    # text = " ".join([res[1] for res in reader.readtext(img_bytes)])

    # Extract fields
    extracted_fields = extract_fields(text)

    return JSONResponse({
        "document_type": document_type,
        "extracted_fields": extracted_fields
    })
