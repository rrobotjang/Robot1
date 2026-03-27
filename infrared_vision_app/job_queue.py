from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from models import (
    AppUser,
    InferenceJobAccepted,
    InferenceJobEnvelope,
    InferenceJobRequest,
    InferenceJobStatus,
    InferenceResponse,
)

try:
    import redis
except Exception:  # noqa: BLE001
    redis = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _dump_model(model) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()


class InferenceJobQueue:
    def submit(self, request: InferenceJobRequest, user: AppUser) -> InferenceJobAccepted:
        raise NotImplementedError

    def get(self, job_id: str) -> InferenceJobEnvelope | None:
        raise NotImplementedError

    def pop_next(self, timeout_seconds: int = 5) -> InferenceJobEnvelope | None:
        raise NotImplementedError

    def mark_processing(self, job_id: str) -> InferenceJobEnvelope | None:
        raise NotImplementedError

    def mark_completed(self, job_id: str, result: InferenceResponse) -> InferenceJobEnvelope | None:
        raise NotImplementedError

    def mark_failed(self, job_id: str, error: str) -> InferenceJobEnvelope | None:
        raise NotImplementedError


class MemoryInferenceJobQueue(InferenceJobQueue):
    def __init__(self) -> None:
        self._jobs: dict[str, InferenceJobEnvelope] = {}
        self._order: list[str] = []

    def submit(self, request: InferenceJobRequest, user: AppUser) -> InferenceJobAccepted:
        job_id = uuid4().hex
        queued_at = _utc_now()
        envelope = InferenceJobEnvelope(
            job_id=job_id,
            status=InferenceJobStatus.QUEUED,
            request=request,
            requested_by=user,
            queued_at=queued_at,
        )
        self._jobs[job_id] = envelope
        self._order.append(job_id)
        return InferenceJobAccepted(job_id=job_id, status=envelope.status, queued_at=queued_at)

    def get(self, job_id: str) -> InferenceJobEnvelope | None:
        return self._jobs.get(job_id)

    def pop_next(self, timeout_seconds: int = 5) -> InferenceJobEnvelope | None:
        _ = timeout_seconds
        while self._order:
            job_id = self._order.pop(0)
            envelope = self._jobs.get(job_id)
            if envelope and envelope.status == InferenceJobStatus.QUEUED:
                return envelope
        return None

    def mark_processing(self, job_id: str) -> InferenceJobEnvelope | None:
        envelope = self._jobs.get(job_id)
        if envelope is None:
            return None
        envelope.status = InferenceJobStatus.PROCESSING
        envelope.started_at = _utc_now()
        self._jobs[job_id] = envelope
        return envelope

    def mark_completed(self, job_id: str, result: InferenceResponse) -> InferenceJobEnvelope | None:
        envelope = self._jobs.get(job_id)
        if envelope is None:
            return None
        envelope.status = InferenceJobStatus.COMPLETED
        envelope.completed_at = _utc_now()
        envelope.result = result
        self._jobs[job_id] = envelope
        return envelope

    def mark_failed(self, job_id: str, error: str) -> InferenceJobEnvelope | None:
        envelope = self._jobs.get(job_id)
        if envelope is None:
            return None
        envelope.status = InferenceJobStatus.FAILED
        envelope.completed_at = _utc_now()
        envelope.error = error
        self._jobs[job_id] = envelope
        return envelope


class RedisInferenceJobQueue(InferenceJobQueue):
    def __init__(self, redis_url: str, queue_name: str, result_ttl_seconds: int = 3600) -> None:
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.queue_name = queue_name
        self.result_ttl_seconds = result_ttl_seconds

    def _key(self, job_id: str) -> str:
        return f"{self.queue_name}:job:{job_id}"

    def _save(self, envelope: InferenceJobEnvelope) -> None:
        key = self._key(envelope.job_id)
        self.client.set(key, json.dumps(_dump_model(envelope), ensure_ascii=False), ex=self.result_ttl_seconds)

    def _load(self, job_id: str) -> InferenceJobEnvelope | None:
        raw = self.client.get(self._key(job_id))
        if not raw:
            return None
        return InferenceJobEnvelope(**json.loads(raw))

    def submit(self, request: InferenceJobRequest, user: AppUser) -> InferenceJobAccepted:
        job_id = uuid4().hex
        queued_at = _utc_now()
        envelope = InferenceJobEnvelope(
            job_id=job_id,
            status=InferenceJobStatus.QUEUED,
            request=request,
            requested_by=user,
            queued_at=queued_at,
        )
        self._save(envelope)
        self.client.rpush(self.queue_name, job_id)
        return InferenceJobAccepted(job_id=job_id, status=envelope.status, queued_at=queued_at)

    def get(self, job_id: str) -> InferenceJobEnvelope | None:
        return self._load(job_id)

    def pop_next(self, timeout_seconds: int = 5) -> InferenceJobEnvelope | None:
        result = self.client.blpop(self.queue_name, timeout=timeout_seconds)
        if not result:
            return None
        _queue_name, job_id = result
        return self._load(job_id)

    def mark_processing(self, job_id: str) -> InferenceJobEnvelope | None:
        envelope = self._load(job_id)
        if envelope is None:
            return None
        envelope.status = InferenceJobStatus.PROCESSING
        envelope.started_at = _utc_now()
        self._save(envelope)
        return envelope

    def mark_completed(self, job_id: str, result: InferenceResponse) -> InferenceJobEnvelope | None:
        envelope = self._load(job_id)
        if envelope is None:
            return None
        envelope.status = InferenceJobStatus.COMPLETED
        envelope.completed_at = _utc_now()
        envelope.result = result
        self._save(envelope)
        return envelope

    def mark_failed(self, job_id: str, error: str) -> InferenceJobEnvelope | None:
        envelope = self._load(job_id)
        if envelope is None:
            return None
        envelope.status = InferenceJobStatus.FAILED
        envelope.completed_at = _utc_now()
        envelope.error = error
        self._save(envelope)
        return envelope


def create_job_queue(queue_backend: str, queue_url: str, queue_name: str, result_ttl_seconds: int) -> InferenceJobQueue:
    if queue_backend == "redis":
        return RedisInferenceJobQueue(queue_url, queue_name, result_ttl_seconds)
    return MemoryInferenceJobQueue()


def run_worker_loop(
    *,
    queue: InferenceJobQueue,
    processor: Callable[[InferenceJobRequest, AppUser], InferenceResponse],
    poll_interval_seconds: float = 0.5,
) -> None:
    while True:
        envelope = queue.pop_next(timeout_seconds=max(1, int(poll_interval_seconds)))
        if envelope is None:
            time.sleep(poll_interval_seconds)
            continue
        queue.mark_processing(envelope.job_id)
        try:
            result = processor(envelope.request, envelope.requested_by)
            queue.mark_completed(envelope.job_id, result)
        except Exception as exc:  # noqa: BLE001
            queue.mark_failed(envelope.job_id, str(exc))
