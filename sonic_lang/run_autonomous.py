#!/usr/bin/env python3
"""
Standalone runner for autonomous object search and approach.

Run this as a SEPARATE process from robot_listener.py.
The robot listener (API server) must already be running on the Pi.

Usage:
    # On your PC or Pi (with robot_listener running on Pi at 192.168.1.100:8080):
    python run_autonomous.py --target bottle
    python run_autonomous.py --target cup --pi-url http://192.168.1.100:8080

    # List available COCO classes:
    python run_autonomous.py --list-classes

    # Scan only (no approach):
    python run_autonomous.py --target bottle --scan-only
"""

import os
import sys
import argparse
import signal

# Ensure we can import from same package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autonomous_controller import (
    AutonomousController,
    HttpRobotClient,
    is_yolo_available,
    YOLODetector,
)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous object search: scan left/center/right, find target, turn and approach until 50%% of frame."
    )
    parser.add_argument(
        "--target",
        type=str,
        help="COCO class name to find (e.g. bottle, person, cup)",
    )
    parser.add_argument(
        "--pi-url",
        type=str,
        default=os.getenv("ROBOT_PI_URL", "http://127.0.0.1:8080"),
        help="Base URL of robot listener (default: ROBOT_PI_URL or http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan 3 positions and report; do not turn or approach",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List available COCO class names and exit",
    )
    args = parser.parse_args()

    if args.list_classes:
        print("Available COCO classes (use --target <name>):")
        for i, name in enumerate(YOLODetector.get_available_classes()):
            print(f"  {name}")
        return 0

    if not args.target:
        parser.error("--target is required (e.g. --target bottle). Use --list-classes to see options.")

    if not is_yolo_available():
        print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
        return 1

    print(f"Connecting to robot at {args.pi_url}")
    client = HttpRobotClient(args.pi_url)
    controller = AutonomousController(client)

    # Set target
    result = controller.set_target(args.target)
    if not result.get("success"):
        print("ERROR:", result.get("error", "Unknown error"))
        if "available_classes" in result:
            print("Available:", result["available_classes"])
        return 1

    print(f"Target set to: {args.target}")

    if args.scan_only:
        print("Running scan only (no approach)...")
        out = controller.scan_once(include_images=False)
        if not out.get("success"):
            print("Scan failed:", out.get("error"))
            return 1
        print("Target found:", out.get("target_found"), "at position:", out.get("target_position"))
        for r in out.get("scan_results", []):
            print(f"  {r['position_name']}: target_found={r['target_found']}")
        return 0

    # Run full autonomous (scan -> turn -> approach)
    def on_stop(sig, frame):
        print("\nStopping autonomous mode...")
        controller.stop()

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    result = controller.start()
    if not result.get("success"):
        print("ERROR:", result.get("error"))
        return 1

    print("Autonomous mode started. Press Ctrl+C to stop.")
    # Block until thread finishes (or Ctrl+C)
    if controller._thread:
        controller._thread.join()

    status = controller.get_status()
    print("Final state:", status.get("state"))
    if status.get("error"):
        print("Error:", status.get("error"))
    return 0


if __name__ == "__main__":
    sys.exit(main())