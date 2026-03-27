from __future__ import annotations

from fastapi import FastAPI

from cv_runtime import run_cv_inference
from models import CVInferenceRequest, CVInferenceResponse


app = FastAPI(title="Infrared CV Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "cv"}


@app.post("/v1/infer", response_model=CVInferenceResponse)
def infer(payload: CVInferenceRequest) -> CVInferenceResponse:
    return run_cv_inference(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("cv_service_app:app", host="0.0.0.0", port=8030, reload=True)
