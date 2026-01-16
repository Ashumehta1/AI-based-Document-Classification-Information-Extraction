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