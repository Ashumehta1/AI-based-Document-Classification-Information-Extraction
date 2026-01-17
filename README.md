##  Document Classification & Information Extraction System
# Overview

    This project is an Intelligent Document Processing (IDP) system that:

Classifies documents into:

    Invoice

    ID Card

    Certificate

    Resume

Extracts key fields using OCR + NLP:

    Name

    Date

    Amount

    ID Number

    Phone

    Email

Exposes predictions through a FastAPI REST API

## The project can be run in two ways:

    1. Using Docker 
          (docker build -t document-ai .
            docker run -p 8000:8000 document-ai)
    2. Without Docker (Python virtual environment)

## Project Structure
nrt_python_assignment/
│
├── app/
│   ├── main.py
│   ├── ocr.py
│   ├── extractor.py
│   ├── vision_classifier.py
│   ├── logger.py
│
├── scripts/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── train_cnn.py
│
├── dataset/
│   ├── invoice/
│   ├── id_card/
│   ├── certificate/
│   └── resume/
│
├── models/
│   └── vision/
│       └── doc_classifier.pth
│
├── requirements.txt
├── Dockerfile
└── README.md


## Setup Instructions

1. **Clone or copy the repository** to your local machine.
2. **Create & activate virtual environment**:

```bash
python -m venv env
env\Scripts\activate 

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
 or 
 {
  "document_type": "certificate",
  "extracted_fields": {
    "name": "Physical Modeling",
    "date": "",
    "amount": "",
    "id_number": "CERTIFICATE",
    "phone": "",
    "email": ""
  }
}
