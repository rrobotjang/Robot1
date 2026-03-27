from __future__ import annotations

from app import build_container
from job_queue import run_worker_loop


def main() -> None:
    container = build_container()
    run_worker_loop(
        queue=container.job_queue,
        processor=container.inference_service.run_job_request,
        poll_interval_seconds=0.5,
    )


if __name__ == "__main__":
    main()
