# stt_module.py
import speech_recognition as sr
import queue

def stt_listener(command_queue: queue.Queue):
    """실시간 음성 입력 감지"""
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("🎤 음성 인식 준비 완료. 말하세요...")

    while True:
        with mic as source:
            audio = r.listen(source, phrase_time_limit=5)
        try:
            cmd = r.recognize_google(audio, language="ko-KR")
            print(f"🗣 입력 감지: {cmd}")
            if cmd in ["종료", "끝내"]:
                command_queue.put("exit")
                break
            command_queue.put(cmd)
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            print("STT 오류:", e)
