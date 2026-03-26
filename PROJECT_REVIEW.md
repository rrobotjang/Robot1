# Project Review (March 26, 2026)

## Scope
This review covers repository structure, implementation maturity, and operational readiness based on the current codebase.

## Overall Assessment
The project is a promising robotics prototype with clear intent (ROS2 + FastAPI + Vite dashboard + orchestration FSM), but it is currently in an early integration stage with duplicated code paths and mock-heavy behavior that will block production deployment without consolidation.

## Follow-up Updates
- Item 5 (authoritative runbook) has now been addressed with `RUNBOOK.md`.

## What Looks Good
- A clear end-to-end target workflow exists: perception → grasp planning → motion → place → UI/state reporting.
- `backend_ros/vla_orchestrator.py` contains a readable FSM model with retry paths and emergency transitions.
- The dashboard app is simple and useful for operations (`/api/start`, `/api/stop`, state and ultrasonic polling).
- Containerized workflows are present for both dev and staged services.

## Key Risks / Gaps

### 1) Multiple overlapping implementations (high maintenance risk)
- There are several parallel app variants (`front_end/app.py`, `front_end/main.py`, `front_end/main(chatgpt-monolithic).py`, `backend_ros/vla_orchestrator.py`) with very similar responsibilities.
- Two frontend directories (`front_end/src` and `vla_frontend/src`) appear duplicated.
- Result: high chance of drift, unclear source-of-truth, and slower debugging.

### 2) Mock/stub behavior in critical robot paths
- Several motion/perception/control functions still use placeholders or simulated outputs.
- This is acceptable for prototyping, but should be explicitly gated behind a `SIMULATION_MODE` switch and clearly surfaced in logs/UI.

### 3) Repository hygiene
- `node_modules` directories are present under `front_end/` and `vla_frontend/` as untracked workspace content.
- This suggests `.gitignore` and repo hygiene standards need strengthening before collaborative scaling.

### 4) API/runtime coupling concerns
- ROS spinning and web server execution are mixed in single processes/threads in prototype scripts.
- This can work in demos but often creates lifecycle and observability issues (shutdown sequencing, deadlocks, partial failures).

### 5) Documentation mismatch
- Root `README.md` is mostly notes and partial tree snapshots, but does not provide a single reliable “how to run in 2026-03” path for developers.
- Multiple compose files refer to directories/services that may not match current top-level structure.

## Recommended Next Steps (Priority Order)
1. **Pick one canonical architecture path** (recommended: orchestrator-centric path around `backend_ros/vla_orchestrator.py`) and deprecate/archive alternate prototypes.
2. **Define simulation vs hardware modes** with explicit env flags and clear startup banner output.
3. **Normalize project layout** (single frontend folder, single backend entrypoint per deployment target).
4. **Add `.gitignore` coverage** for JS/Python artifacts and enforce via CI.
5. **Publish one authoritative runbook**:
   - local dev steps,
   - container dev steps,
   - required ROS topics/services,
   - expected health checks.
6. **Introduce minimal CI checks** (lint + type checks + smoke tests) to prevent drift.

## Suggested Milestone Definition
A practical “v0.2 stable integration” milestone would include:
- one orchestrator entrypoint,
- one frontend,
- no duplicated runtime scripts in active path,
- explicit simulation mode,
- reproducible startup via one compose profile,
- basic automated checks passing in CI.
