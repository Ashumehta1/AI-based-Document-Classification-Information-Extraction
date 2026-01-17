# Base image
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip first (VERY IMPORTANT)
RUN pip install --upgrade pip

# Install Python dependencies (without torch)
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version explicitly
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
