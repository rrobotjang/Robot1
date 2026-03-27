from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable
from urllib import error, request

from models import (
    KnowledgeDocument,
    SiteProfile,
    UserRole,
    VectorSearchResponse,
    VectorSearchResult,
    VectorStoreStatus,
)
from sample_data import build_knowledge_documents


EMBED_DIMENSION = 64


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣_]+", text.lower())


def _embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBED_DIMENSION
    for token in _tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % EMBED_DIMENSION
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[idx] += sign * (1.0 + min(len(token), 12) / 12.0)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _build_snippet(doc: KnowledgeDocument, query: str) -> str:
    body = doc.body.strip()
    if not body:
        return ""

    tokens = _tokenize(query)
    lowered = body.lower()
    for token in tokens:
        idx = lowered.find(token.lower())
        if idx >= 0:
            start = max(0, idx - 40)
            end = min(len(body), idx + 96)
            snippet = body[start:end].strip()
            return snippet if start == 0 else f"...{snippet}"
    return body[:140].strip()


def _doc_visible(doc: KnowledgeDocument, profile: SiteProfile | None, role: UserRole) -> bool:
    if doc.allowed_roles and role not in doc.allowed_roles:
        return False
    if doc.profile is None:
        return True
    return profile == doc.profile


class BaseVectorStore:
    backend = "memory"

    def upsert_documents(self, documents: Iterable[KnowledgeDocument]) -> None:
        raise NotImplementedError

    def search(
        self,
        query: str,
        profile: SiteProfile | None,
        role: UserRole,
        limit: int,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError


@dataclass
class _StoredDoc:
    document: KnowledgeDocument
    vector: list[float]


class InMemoryVectorStore(BaseVectorStore):
    backend = "memory"

    def __init__(self) -> None:
        self._documents: Dict[str, _StoredDoc] = {}

    def upsert_documents(self, documents: Iterable[KnowledgeDocument]) -> None:
        for doc in documents:
            source = " ".join([doc.title, doc.body, " ".join(doc.tags)])
            self._documents[doc.id] = _StoredDoc(document=doc, vector=_embed_text(source))

    def search(
        self,
        query: str,
        profile: SiteProfile | None,
        role: UserRole,
        limit: int,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            return []

        query_vector = _embed_text(query)
        scored = []
        for stored in self._documents.values():
            doc = stored.document
            if not _doc_visible(doc, profile, role):
                continue
            score = _cosine_similarity(query_vector, stored.vector)
            scored.append(
                VectorSearchResult(
                    id=doc.id,
                    title=doc.title,
                    snippet=_build_snippet(doc, query),
                    score=round(float(score), 4),
                    tags=doc.tags,
                    profile=doc.profile,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


class QdrantVectorStore(BaseVectorStore):
    backend = "qdrant"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:6333",
        collection: str = "infrared_vision_knowledge",
        timeout: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        req = request.Request(
            f"{self.base_url}{path}",
            method=method,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def _ensure_collection(self) -> None:
        self._request(
            "PUT",
            f"/collections/{self.collection}",
            {
                "vectors": {
                    "size": EMBED_DIMENSION,
                    "distance": "Cosine",
                }
            },
        )

    def upsert_documents(self, documents: Iterable[KnowledgeDocument]) -> None:
        self._ensure_collection()
        points = []
        for doc in documents:
            vector = _embed_text(" ".join([doc.title, doc.body, " ".join(doc.tags)]))
            points.append(
                {
                    "id": doc.id,
                    "vector": vector,
                    "payload": {
                        "title": doc.title,
                        "body": doc.body,
                        "tags": doc.tags,
                        "profile": doc.profile.value if doc.profile else None,
                        "allowed_roles": [role.value for role in doc.allowed_roles],
                    },
                }
            )

        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {"points": points},
        )

    def search(
        self,
        query: str,
        profile: SiteProfile | None,
        role: UserRole,
        limit: int,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            return []

        response = self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            {
                "vector": _embed_text(query),
                "limit": max(limit * 3, 8),
                "with_payload": True,
            },
        )

        results = []
        for item in response.get("result", []):
            payload = item.get("payload", {})
            doc = KnowledgeDocument(
                id=str(item.get("id")),
                title=payload.get("title", ""),
                body=payload.get("body", ""),
                tags=payload.get("tags", []),
                profile=payload.get("profile"),
                allowed_roles=payload.get("allowed_roles", []),
            )
            if not _doc_visible(doc, profile, role):
                continue
            results.append(
                VectorSearchResult(
                    id=doc.id,
                    title=doc.title,
                    snippet=_build_snippet(doc, query),
                    score=round(float(item.get("score", 0.0)), 4),
                    tags=doc.tags,
                    profile=doc.profile,
                )
            )
            if len(results) >= limit:
                break
        return results


class KnowledgeVectorService:
    def __init__(
        self,
        *,
        backend_name: str = "memory",
        qdrant_url: str = "http://127.0.0.1:6333",
        qdrant_collection: str = "infrared_vision_knowledge",
        qdrant_timeout: float = 3.0,
    ) -> None:
        backend_name = backend_name.lower()
        self._fallback = InMemoryVectorStore()
        self._primary: BaseVectorStore
        if backend_name == "qdrant":
            self._primary = QdrantVectorStore(
                base_url=qdrant_url,
                collection=qdrant_collection,
                timeout=qdrant_timeout,
            )
        else:
            self._primary = self._fallback

        self._documents = build_knowledge_documents()
        self._fallback.upsert_documents(self._documents)
        if self._primary is not self._fallback:
            try:
                self._primary.upsert_documents(self._documents)
            except Exception:
                self._primary = self._fallback

    def search(
        self,
        query: str,
        profile: SiteProfile | None,
        role: UserRole,
        limit: int = 4,
    ) -> VectorSearchResponse:
        try:
            results = self._primary.search(query, profile, role, limit)
            backend = self._primary.backend
        except (error.URLError, TimeoutError, ValueError):
            results = self._fallback.search(query, profile, role, limit)
            backend = self._fallback.backend

        return VectorSearchResponse(
            backend=backend,
            query=query,
            results=results,
        )

    def status(self) -> VectorStoreStatus:
        return VectorStoreStatus(
            backend=self._primary.backend,
            document_count=len(self._documents),
        )
