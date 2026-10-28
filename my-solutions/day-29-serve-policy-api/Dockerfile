FROM python:3.11-slim
WORKDIR /app
COPY my-solutions/day-29-serve-policy-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
COPY my-solutions/day-29-serve-policy-api/app.py .
ENV PYTHONPATH=/app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]