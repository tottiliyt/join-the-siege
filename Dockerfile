FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Install Tesseract OCR for image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy application code (filtered by .dockerignore)
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DEBUG=True

# Expose port
EXPOSE 8080

# Run the application with Gunicorn
CMD exec gunicorn --bind :$PORT --workers $(nproc) --threads 8 --timeout 0 src.app:app
