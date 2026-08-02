FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY outputs/ outputs/

EXPOSE 8501

CMD [
    "streamlit", "run", "app/app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0",
    "--browser.gatherUsageStats=false",
]
