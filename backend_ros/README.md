Quick notes & troubleshooting

This is a production-style orchestrator but still a prototype: tune thresholds (DEFAULT_ULTRA_SLOW, DEFAULT_ULTRA_STOP) to match your gripper geometry.

If MoveItClient, GripperInterface, or PcDetector are not found, the node runs in mock mode for those actions (sleep/log) — safe for dry-run testing.

For real robot use: ensure MoveIt planning group name matches (e.g., "manipulator"), the Doosan driver is running, RealSense publishes pointcloud, and ultrasonic topics publish meters as Float32.

Add a hardware E-STOP bound to a real safety node for industrial deployments. Software emergency is a convenience stop, not a substitute for hardware e-stop.
