FROM python:3.10-slim

WORKDIR /app

COPY requirements.haas.txt .
RUN pip install --no-cache-dir -r requirements.haas.txt

COPY . /app

CMD ["python", "-u", "runpod_handler.py"]
