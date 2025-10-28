# main.py
import threading, queue
from clova_module import clova_chat, clova_tts
from robot_module import model_forward, execute_robot_action, robot_busy, stop_event
from command_classifier import classify_command
from stt_module import stt_listener

def controller(command_queue: queue.Queue):
    """입력 명령 분류 및 제어 스레드"""
    while True:
        cmd = command_queue.get()
        if cmd == "exit":
            print("[System] 종료합니다.")
            break

        ctype = classify_command(cmd)

        if ctype == "STOP":
            stop_event.set()
            print("[System] 긴급 정지 명령 수신")
            clova_tts("멈출게요.")
            continue

        elif ctype == "ROBOT":
            if robot_busy.is_set():
                print("[System] 로봇이 이미 동작 중입니다.")
                clova_tts("지금은 이미 움직이고 있어요.")
                continue
            robot_busy.set()
            pose = model_forward(cmd)
            clova_tts("알겠어요. 움직일게요.")
            execute_robot_action(pose)
            clova_tts("동작 완료.")

        elif ctype == "CHAT":
            if robot_busy.is_set():
                reply = "지금 일하는 중이에요. 잠시만요."
            else:
                reply = clova_chat(cmd)
            print("CLOVA🤖:", reply)
            clova_tts(reply)

if __name__ == "__main__":
    q = queue.Queue()
    t1 = threading.Thread(target=stt_listener, args=(q,), daemon=True)
    t2 = threading.Thread(target=controller, args=(q,), daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()
