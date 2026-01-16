# # # from fastapi import FastAPI, UploadFile, File
# # # from PIL import Image
# # # import io

# # # from app.ocr import extract_text
# # # from app.extractor import extract_fields
# # # from app.vision_classifier import classify_image
# # # from app.logger import init_db   # <-- import init_db
# # # from app.logger import log_prediction 

# # # # Initialize DB at startup
# # # init_db()  # <-- call it here

# # # app = FastAPI(title="Document AI")

# # # @app.get("/health")
# # # def health():
# # #     return {"status": "ok"}

# # # @app.post("/analyze")
# # # async def analyze(file: UploadFile = File(...)):
# # #     img = Image.open(io.BytesIO(await file.read())).convert("RGB")

# # #     doc_type = classify_image(img)
# # #     text = extract_text(img)
# # #     fields = extract_fields(text)

# # #     return {
# # #         "document_type": doc_type,
# # #         "extracted_fields": fields
# # #     }
# # from fastapi import FastAPI, UploadFile, File
# # from PIL import Image
# # import io

# # from app.ocr import extract_text
# # from app.extractor import extract_fields
# # from app.vision_classifier import classify_image
# # from app.logger import init_db, log_prediction  # import both

# # # Initialize DB at startup
# # init_db()

# # app = FastAPI(title="Document AI")

# # @app.get("/health")
# # def health():
# #     return {"status": "ok"}

# # @app.post("/analyze")
# # async def analyze(file: UploadFile = File(...)):
# #     filename = file.filename  # save original filename
# #     img = Image.open(io.BytesIO(await file.read())).convert("RGB")

# #     # Step 1: classify document
# #     doc_type = classify_image(img)

# #     # Step 2: OCR & extract fields
# #     text = extract_text(img)
# #     fields = extract_fields(text)

# #     # Step 3: log prediction into DB
# #     log_prediction(filename, doc_type, fields)

# #     return {
# #         "document_type": doc_type,
# #         "extracted_fields": fields
# #     }

# from fastapi import FastAPI, UploadFile, File
# from PIL import Image
# import io

# from app.ocr import extract_text
# from app.extractor import extract_fields
# from app.vision_classifier import classify_image
# from app.logger import init_db, log_prediction

# # PDF support
# from pdf2image import convert_from_bytes

# # Initialize DB
# init_db()

# app = FastAPI(title="Document AI")

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/analyze")
# async def analyze(file: UploadFile = File(...)):
#     filename = file.filename
#     content = await file.read()
#     images = []

#     # ---------------------------
#     # Handle Images
#     # ---------------------------
#     if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         images.append(Image.open(io.BytesIO(content)).convert("RGB"))

#     # ---------------------------
#     # Handle PDFs
#     # ---------------------------
#     elif filename.lower().endswith(".pdf"):
#         # convert PDF pages to images
#         images = convert_from_bytes(content)  # optionally: poppler_path="C:/poppler-23.05.0/Library/bin"

#     else:
#         return {"error": "Unsupported file type. Only images and PDFs are allowed."}

#     # ---------------------------
#     # Process each image / PDF page
#     # ---------------------------
#     results = []
#     for img in images:
#         doc_type = classify_image(img)
#         text = extract_text(img)
#         fields = extract_fields(text)

#         # log each page
#         log_prediction(filename, doc_type, fields)

#         results.append({
#             "document_type": doc_type,
#             "extracted_fields": fields
#         })

#     # If single page, return first element, else return list
#     return results[0] if len(results) == 1 else results

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from pdf2image import convert_from_bytes

from app.ocr import extract_text
from app.extractor import extract_fields
from app.vision_classifier import classify_image
from app.logger import init_db, log_prediction

# Initialize DB
init_db()

app = FastAPI(title="Document AI")

POPPLER_PATH = r"C:\poppler-23.05.0\Library\bin"  # <-- your Poppler bin path

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    # Check file type
    if file.filename.lower().endswith(".pdf"):
        # Convert PDF pages to images
        images = convert_from_bytes(content, poppler_path=POPPLER_PATH)
        # For now, just process first page (you can loop for multi-page later)
        img = images[0]
    else:
        img = Image.open(io.BytesIO(content)).convert("RGB")

    # Document classification and field extraction
    doc_type = classify_image(img)
    text = extract_text(img)
    fields = extract_fields(text)

    # Optionally log prediction
    log_prediction(file.filename, doc_type, fields)

    return {
        "document_type": doc_type,
        "extracted_fields": fields
    }
