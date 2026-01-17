# 1. Base Python image
FROM python:3.10-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Install system dependencies (Tesseract OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (for faster rebuilds)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy entire project
COPY . .

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
