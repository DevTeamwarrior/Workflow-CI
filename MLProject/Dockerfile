# 1. Menggunakan base image Python resmi versi slim agar ukuran container ringan
FROM python:3.12-slim

# 2. Mengatur environment variable agar Python tidak menulis file .pyc (bytecode) 
# dan output log langsung dialirkan ke terminal tanpa di-buffer (real-time)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Menetapkan direktori kerja utama di dalam kontainer Docker
WORKDIR /app

# 4. Menyalin berkas dependensi (conda.yaml) ke dalam kontainer
COPY conda.yaml /app/conda.yaml

# 5. Memperbarui pip dan menginstal semua pustaka (libraries) yang dibutuhkan proyek
# Menggunakan opsi --no-cache-dir agar kontainer tidak menyimpan sampah instalasi
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow==2.16.1 \
    mlflow==2.19.0 \
    dagshub \
    matplotlib

# 6. Menyalin skrip kode pemodelan utama ke dalam kontainer
COPY modelling.py /app/modelling.py

# 7. Menyalin berkas MLProject sebagai standar komponen MLflow Project
COPY MLProject /app/MLProject

# 8. Instruksi default untuk menjalankan retraining model saat kontainer Docker dinyalakan
CMD ["python", "modelling.py"]