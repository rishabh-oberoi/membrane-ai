FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "agniscient.main:app", "--host", "0.0.0.0", "--port", "8000"]
