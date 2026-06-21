# ============================================
# AI Model & Dataset Management Platform
# Multi-stage build optimised for ML dependencies
# ============================================

FROM python:3.12-slim

WORKDIR /app

# System dependencies (libgomp1 needed by scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose Flask dev server port
# (Use gunicorn in production for better performance)
EXPOSE 5000

# Health check via platform health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "run.py"]
