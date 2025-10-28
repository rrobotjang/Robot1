# clova_module.py
import requests, json, playsound
from pathlib import Path
from config import CHAT_URL, TTS_URL, CHAT_KEY, TTS_ID, TTS_SECRET

chat_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CHAT_KEY}"
}
tts_headers = {
    "X-NCP-APIGW-API-KEY-ID": TTS_ID,
    "X-NCP-APIGW-API-KEY": TTS_SECRET
}

def clova_chat(message: str) -> str:
    """자유 대화용 CLOVA Chat API"""
    payload = {"messages": [{"role": "user", "content": message}]}
    try:
        resp = requests.post(CHAT_URL, headers=chat_headers, json=payload)
        data = resp.json()
        return data["result"]["message"]["content"]
    except Exception:
        return "지금은 대답할 수 없어요."

def clova_tts(text: str, speaker="nara"):
    """TTS로 텍스트를 음성으로 변환 후 재생"""
    payload = {"speaker": speaker, "text": text, "format": "mp3"}
    path = Path("reply.mp3")
    resp = requests.post(TTS_URL, headers=tts_headers, data=payload)
    with open(path, "wb") as f:
        f.write(resp.content)
    playsound.playsound(str(path))
