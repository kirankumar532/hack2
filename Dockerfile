FROM python:3.8.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "model_app:app", "--host", "0.0.0.0", "--port", "80"]

