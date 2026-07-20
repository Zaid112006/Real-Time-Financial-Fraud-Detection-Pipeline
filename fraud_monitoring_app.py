from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Fraud Detection Monitoring API"
)

# Enable Prometheus monitoring
Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {"message": "Fraud Detection Monitoring API Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}