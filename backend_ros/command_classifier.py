# command_classifier.py
def classify_command(cmd: str) -> str:
    stop_words = ["멈춰", "그만", "중단"]
    robot_words = ["잡아", "집어", "놓아", "옮겨", "그립", "이동", "들어"]
    if any(w in cmd for w in stop_words):
        return "STOP"
    elif any(w in cmd for w in robot_words):
        return "ROBOT"
    else:
        return "CHAT"

