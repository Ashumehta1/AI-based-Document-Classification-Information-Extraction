# app/ocr.py

from PIL import Image
import easyocr
import pytesseract

# Set the path to the installed tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# text extraction
def extract_text(image: Image.Image) -> str:
    """
    Extract text from an image using pytesseract
    """
    text = pytesseract.image_to_string(image)
    return text

# EasyOCR-based text extraction 
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_easyocr(image: Image.Image) -> str:
    """
    Extract text using EasyOCR
    """
    results = reader.readtext(image)
    text = " ".join([res[1] for res in results])
    return text
