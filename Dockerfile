# Backend (Python + OpenCV + ONNX). Netlify'dagi frontend /api/* so'rovlarini shu serverga yuboradi.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DATA_DIR=/data SETTINGS_FILE=/data/settings.json MODEL_DIR=/app/models \
    SECURE_COOKIES=1 BEHIND_PROXY=1 PORT=8080

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Modellar image ichiga build paytida yuklanadi (~300 MB)
COPY download_models.py .
RUN python download_models.py

COPY recognizer.py attributes.py auth.py server.py ./
COPY web ./web

# /data - doimiy disk (baza, sozlamalar, sessiya kaliti). Hosting'da volume ulang!
VOLUME ["/data"]
EXPOSE 8080

# 1 ta worker: tanish holati (kuzatuv, ro'yxatga olish) xotirada saqlanadi; parallellik - threadlar orqali
CMD gunicorn server:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT}
