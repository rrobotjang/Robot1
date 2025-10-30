# Robotics-Project-VLA
backend_ros/
├─ vla_perception/
│  ├─ package.xml
│  ├─ setup.cfg
│  ├─ setup.py
│  └─ vla_perception/
│     └─ node_perception.py
├─ vla_control/
│  ├─ package.xml
│  ├─ setup.cfg
│  ├─ setup.py
│  └─ vla_control/
│     └─ gripper_node.py
├─ vla_motion/
│  ├─ package.xml
│  ├─ setup.cfg
│  ├─ setup.py
│  ├─ launch/
│  │  └─ vla_system_launch.py
│  └─ vla_motion/
│     └─ orchestrator.py   # main orchestration node (pick&place workflow)
└─ README.md

# Monolithic
robotics_project/
├── main.py           # FastAPI + rclpy 통합 서버
├── frontend.py       # Gradio UI (FastAPI mount)
├── ros_bridge.py     # ROS2 서비스 호출 로직
├── tts_feedback.py   # OpenAI Realtime API 호출
└── requirements.txt

#실행
uvicorn main:app --reload
