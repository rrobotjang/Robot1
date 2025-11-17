FROM ros:humble

# System
RUN apt update && apt install -y python3-pip python3-colcon-common-extensions

# Backend FastAPI
RUN pip3 install fastapi uvicorn

# Put orchestrator code
WORKDIR /app
COPY orchestrator_node.py /app/
COPY setup.py /app/
COPY vla_ws /app/vla_ws

# Install robot packages
WORKDIR /app/vla_ws
RUN rosdep install --from-paths src -y
RUN colcon build
RUN echo "source /app/vla_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /app
EXPOSE 8000

CMD ["bash", "-c", "source /app/vla_ws/install/setup.bash && uvicorn orchestrator_node:app --host 0.0.0.0 --port 8000"]
