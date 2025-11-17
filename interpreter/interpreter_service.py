# interpreter/interpreter_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import re

app = FastAPI(title="VLA Interpreter")

FRONTEND_EXECUTE_URL = os.getenv("FRONTEND_EXECUTE_URL", "http://localhost:8000/execute")

class RawCmd(BaseModel):
    text: str
    user_id: str | None = None

class StructuredCmd(BaseModel):
    intent: str
    action: str
    params: dict

def rule_based_parse(text: str) -> StructuredCmd:
    t = text.strip().lower()
    # 간단 룰 예시 — 필요시 확장
    if "home" in t or "홈" in t:
        return StructuredCmd(intent="motion", action="move_joint", params={"pose":"home"})
    if "pick" in t or "픽업" in t or "take" in t:
        # ex: "빨간 캔 픽업"
        color = "red" if "red" in t or "빨간" in t or "빨강" in t else "unknown"
        return StructuredCmd(intent="manipulation", action="pick_object", params={"object":"can","color":color})
    if m := re.search(r"move to x=(?P<x>[-\d.]+),y=(?P<y>[-\d.]+)", t):
        return StructuredCmd(intent="motion", action="move_line", params={"x":float(m.group("x")),"y":float(m.group("y"))})
    # default
    return StructuredCmd(intent="unknown", action="noop", params={})

@app.post("/interpret")
def interpret(cmd: RawCmd):
    # TODO: call OpenAI here if available (placeholder)
    use_openai = os.getenv("USE_OPENAI","false").lower() in ("1","true","yes")
    if use_openai:
        # Placeholder: implement real OpenAI call here
        # For now fall back to rules
        structured = rule_based_parse(cmd.text)
    else:
        structured = rule_based_parse(cmd.text)

    # forward to frontend
    resp = requests.post(FRONTEND_EXECUTE_URL, json=structured.dict(), timeout=10)
    return {"ok": True, "structured": structured.dict(), "frontend_response_status": resp.status_code, "frontend_response": resp.json()}
