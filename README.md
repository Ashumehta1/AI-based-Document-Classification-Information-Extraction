# Document Classification & Information Extraction System

## Overview
This project is an intelligent document processing pipeline capable of:

1. **Classifying documents** into types: Invoice, ID Card, Certificate, Resume.
2. **Extracting key fields** like Name, Date, Amount, ID Number, Phone, Email using OCR and regex-based NLP.
3. **Serving predictions via a REST API** built with FastAPI.

## Folder Structure

nrt_python_assignment/
│
├── app/
│ ├── main.py
│ ├── ocr.py
│ ├── extractor.py
│ ├── vision_classifier.py
│ ├── logger.py
│
├── scripts/
│ ├── init.py
│ ├── generate_dataset.py
│ ├── train_cnn.py
│
├── dataset/ # Document images for training
│ ├── invoice/
│ ├── id_card/
│ ├── certificate/
│ └── resume/
│
├── models/
│ └── vision/
│ └── doc_classifier.pth
│
├── env/ # Virtual environment
├── requirements.txt
└── README.md


## Setup Instructions

1. **Clone or copy the repository** to your local machine.
2. **Create & activate virtual environment**:

```bash
python -m venv env
env\Scripts\activate  # Windows
# or
source env/bin/activate  # Linux/Mac

## pip install -r requirements.txt
pip install -r requirements.txt

## Train CNN (optional)
python scripts/train_cnn.py

## Run FastAPI server
uvicorn app.main:app --reload

## Open Swagger UI
Navigate to http://127.0.0.1:8000/docs to test endpoints.

## API Endpoints
# Health Check
GET /health
Response: {"status": "ok"}

## Document Analysis
POST /analyze
Accepts: Image file (PNG, JPG)
Returns: JSON with document type and extracted fields

##Sample Response:
	
Response body
Download
{
  "document_type": "id_card",
  "extracted_fields": {
    "name": "Sonu Kumar",
    "date": "20/09/1984",
    "amount": "",
    "id_number": "F890OB",
    "phone": "",
    "email": ""
  }
}

0r

	
Response body
Download
{
  "document_type": "resume",
  "extracted_fields": {
    "name": "Data Science",
    "date": "",
    "amount": "",
    "id_number": "ASHISH",
    "phone": "9043458873",
    "email": "ashi76ri08mehta@gmail.com"
  }
}

or 	
Response body
Download
{
  "document_type": "resume",
  "extracted_fields": {
    "name": "Python Developer",
    "date": "",
    "amount": "",
    "id_number": "ASHISH",
    "phone": "7004333373",
    "email": "ashisj08mehta@gmail.com"
  }
}

