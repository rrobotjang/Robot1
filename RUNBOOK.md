# VLA System Runbook (Authoritative)

_Last updated: March 26, 2026 (UTC)_

This runbook defines one **canonical developer workflow** for this repository.

---

## 1) Canonical Runtime Path

Use the **orchestrator backend** as the source of truth:

- Backend entrypoint: `backend_ros/vla_orchestrator.py`
- HTTP API (FastAPI): port `8080`
- Key endpoint: `POST /voice`

> Note: Other scripts in this repo are kept for prototype history, but this runbook standardizes on the orchestrator path above.

---

## 2) Prerequisites

- Python 3.10+
- ROS2 Humble environment available (`/opt/ros/humble/setup.bash`)
- Optional hardware integrations:
  - Depth camera publishing point cloud
  - Ultrasonic sensor publishing `Float32` (meters)
  - MoveIt client and gripper interface packages

If MoveIt/perception/gripper wrappers are unavailable, orchestrator behavior falls back to mock/safe dry-run flows.

---

## 3) Local Development Startup

From repository root:

```bash
# 1) Python deps (minimum baseline)
pip install -r requirements.txt

# 2) Source ROS2
source /opt/ros/humble/setup.bash

# 3) Start backend orchestrator
python backend_ros/vla_orchestrator.py
```

Expected backend bind:
- `http://0.0.0.0:8080`

---

## 4) API Contract (Current)

### `POST /voice`
Request body:

```json
{
  "text": "pick red can"
}
```

Current command routing:
- `"pick"` or `"픽업"` → starts async pick/place workflow
- `"stop"` or `"정지"` or `"멈춰"` → triggers emergency stop path
- otherwise → unknown command response

Example test calls:

```bash
# start workflow
curl -s -X POST http://localhost:8080/voice \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick"}'

# emergency stop
curl -s -X POST http://localhost:8080/voice \
  -H 'Content-Type: application/json' \
  -d '{"text":"stop"}'
```

---

## 5) ROS Topics Required for Full Hardware Flow

### Subscribed by orchestrator
- `/camera/depth/color/points` (`sensor_msgs/msg/PointCloud2`)
- `/ultrasonic/distance` (`std_msgs/msg/Float32`)
- `/vla/emergency_stop` (`std_msgs/msg/Bool`)

### Published by orchestrator
- `/vla/state` (`std_msgs/msg/String`)

Recommended quick checks:

```bash
ros2 topic list
ros2 topic echo /vla/state
ros2 topic hz /camera/depth/color/points
ros2 topic hz /ultrasonic/distance
```

---

## 6) Health Checks

### Backend health
```bash
curl -s -X POST http://localhost:8080/voice \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick"}'
```
Expected: JSON with `"ok": true` and start message.

### Emergency path health
```bash
curl -s -X POST http://localhost:8080/voice \
  -H 'Content-Type: application/json' \
  -d '{"text":"stop"}'
```
Expected: JSON with `"ok": true` and emergency message.

### Orchestrator state telemetry
```bash
ros2 topic echo /vla/state
```
Expected states include `IDLE`, `BUSY`, `ERROR`, `EMERGENCY`.

---

## 7) Frontend Status

A Vite dashboard exists (`vla_frontend` / `front_end`), but current route assumptions (`/api/start`, `/api/stop`, `/api/state`, `/api/ultrasonic`) do not match the orchestrator API contract (`POST /voice`) yet.

Until API harmonization is completed, backend verification should be done directly with `curl` and ROS topic checks.

---

## 8) Troubleshooting

- **Backend starts but no picks execute**: verify point cloud stream exists and is fresh.
- **Ultrasonic-guided steps never complete**: verify `/ultrasonic/distance` publishes meters and updates regularly.
- **Emergency appears stuck**: publish `False` on `/vla/emergency_stop` and confirm `/vla/state` returns to `IDLE`.
- **MoveIt errors**: confirm correct planning group (`manipulator`) and driver stack availability.

---

## 9) Change Control for This Runbook

If runtime contracts or entrypoints change:
1. Update this `RUNBOOK.md` in the same PR.
2. Include exact endpoint/topic changes in PR description.
3. Re-run the health checks in section 6 and paste results in PR notes.
