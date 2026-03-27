from __future__ import annotations

import json
from urllib import request

from cv_runtime import run_cv_inference
from models import CVInferenceRequest, CVInferenceResponse


class CVInferenceClient:
    def infer(self, payload: CVInferenceRequest) -> CVInferenceResponse:
        raise NotImplementedError


def _dump_model(model) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()


class LocalCVInferenceClient(CVInferenceClient):
    def infer(self, payload: CVInferenceRequest) -> CVInferenceResponse:
        return run_cv_inference(payload)


class RemoteCVInferenceClient(CVInferenceClient):
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def infer(self, payload: CVInferenceRequest) -> CVInferenceResponse:
        req = request.Request(
            f"{self.base_url}/v1/infer",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(_dump_model(payload), ensure_ascii=False).encode("utf-8"),
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
        return CVInferenceResponse(**json.loads(raw))
