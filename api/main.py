"""FastAPI application with SSE streaming."""
from fastapi import FastAPI

app = FastAPI(title="PCOS Care Navigator API")

@app.get("/")
def read_root():
    return {"status": "ok", "app": "PCOS Care Navigator"}
