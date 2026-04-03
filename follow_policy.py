"""
Shared human-follow control and person selection (no YOLO / serial / camera).
Used by human_following.py (local) and pc_follow_server.py (GPU offload).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class FollowConfig:
    target_class: str = "person"
    min_confidence: float = 0.4
    forward_speed: int = 150
    turn_speed_min: int = 115
    turn_speed_max: int = 200
    center_tolerance: float = 0.05
    stop_area_ratio: float = 0.35
    lost_frames_before_search: int = 20
    search_turn_duration: float = 0.25
    search_alternate_every: int = 3
    smooth_alpha: float = 0.35
    turn_pulse_sec: float = 0.0


@dataclass
class FollowState:
    lost_frames: int = 0
    search_phase: int = 0
    x_dev_smooth: float = 0.0
    prev_track_id: Optional[int] = None
    prev_center: Optional[Tuple[float, float]] = None
    search_active_until: float = 0.0
    search_current_cmd: Optional[str] = None
    frame_idx: int = 0
    last_stale_box: Any = None
    last_stale_conf: float = 0.0


@dataclass
class FollowStepResult:
    cmd: str
    status: str
    bbox: Optional[Tuple[int, int, int, int]]
    conf: float
    track_id: Optional[int]
    x_dev_smooth: float
    area_ratio: float


def box_center_xyxy(box: Any) -> Tuple[float, float]:
    if hasattr(box, "cpu"):
        box = box.cpu().numpy()
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def xyxy_to_int(box: Any) -> Tuple[int, int, int, int]:
    if hasattr(box, "cpu"):
        box = box.cpu().numpy()
    flat = box.reshape(-1) if hasattr(box, "reshape") else box
    x1, y1, x2, y2 = (int(float(flat[i])) for i in range(4))
    return x1, y1, x2, y2


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def select_person_box(
    results,
    model_names,
    cfg: FollowConfig,
    prev_track_id: Optional[int],
    prev_center: Optional[Tuple[float, float]],
):
    """
    Prefer previous track id, else closest box center to prev_center, else highest conf.
    Returns (box, conf, center_xy, track_id_or_none).
    """
    candidates: List[Tuple[Any, float, Optional[int]]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes
        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        tid_tensor = getattr(boxes, "id", None)
        if tid_tensor is not None:
            tids = tid_tensor.cpu().numpy().astype(int).tolist()
        else:
            tids = [None] * len(boxes)
        for i in range(len(boxes)):
            c = float(confs[i])
            ci = int(clss[i])
            label = model_names[ci]
            if label != cfg.target_class or c < cfg.min_confidence:
                continue
            box = xyxy[i]
            tid = tids[i] if i < len(tids) else None
            candidates.append((box, c, tid))

    if not candidates:
        return None, None, None, None

    if prev_track_id is not None:
        for box, c, tid in candidates:
            if tid is not None and int(tid) == int(prev_track_id):
                cx, cy = box_center_xyxy(box)
                return box, float(c), (cx, cy), int(tid)

    if prev_center is not None:
        best_box, best_c, best_tid = None, -1.0, None
        best_d = math.inf
        for box, c, tid in candidates:
            cx, cy = box_center_xyxy(box)
            d = dist2((cx, cy), prev_center)
            if d < best_d:
                best_d = d
                best_box, best_c, best_tid = box, c, tid
        if best_box is not None:
            cx, cy = box_center_xyxy(best_box)
            tid_i = int(best_tid) if best_tid is not None else None
            return best_box, float(best_c), (cx, cy), tid_i

    best_box, best_c, best_tid = None, -1.0, None
    for box, c, tid in candidates:
        if c > best_c:
            best_c = c
            best_box = box
            best_tid = tid
    cx, cy = box_center_xyxy(best_box)
    tid_i = int(best_tid) if best_tid is not None else None
    return best_box, float(best_c), (cx, cy), tid_i


def follow_control_step(
    state: FollowState,
    cfg: FollowConfig,
    *,
    now: float,
    fw: int,
    fh: int,
    person_found: bool,
    best_box: Any,
    best_conf: float,
    tid_pick: Optional[int],
    run_infer: bool,
) -> FollowStepResult:
    """
    One control iteration: updates FollowState in place, returns command and telemetry.
    """
    frame_area = float(fw * fh)
    bbox_out: Optional[Tuple[int, int, int, int]] = None
    area_ratio_out = 0.0
    track_out: Optional[int] = tid_pick if person_found else None

    if person_found:
        state.lost_frames = 0
        state.search_active_until = 0.0
        state.search_current_cmd = None
        x1, y1, x2, y2 = xyxy_to_int(best_box)
        bbox_out = (x1, y1, x2, y2)

        if run_infer and tid_pick is not None:
            state.prev_track_id = tid_pick
        state.prev_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        box_area = (x2 - x1) * (y2 - y1)
        area_ratio_out = box_area / frame_area

        obj_center_x = (x1 + x2) / 2
        x_deviation = (obj_center_x / fw) - 0.5
        state.x_dev_smooth = (
            cfg.smooth_alpha * x_deviation
            + (1.0 - cfg.smooth_alpha) * state.x_dev_smooth
        )
        x_use = state.x_dev_smooth

        if abs(x_use) > cfg.center_tolerance:
            turn_strength = min(abs(x_use) * 400, cfg.turn_speed_max)
            turn_speed = max(int(turn_strength), cfg.turn_speed_min)
            cmd_turn = f"L:{turn_speed}" if x_use < 0 else f"R:{turn_speed}"

            if cfg.turn_pulse_sec <= 0:
                return FollowStepResult(
                    cmd=cmd_turn,
                    status="TURN LEFT" if x_use < 0 else "TURN RIGHT",
                    bbox=bbox_out,
                    conf=best_conf,
                    track_id=track_out,
                    x_dev_smooth=x_use,
                    area_ratio=area_ratio_out,
                )
            period = cfg.turn_pulse_sec * 2.0
            phase_t = now % period if period > 0 else 0.0
            if phase_t < cfg.turn_pulse_sec:
                return FollowStepResult(
                    cmd=cmd_turn,
                    status="TURN LEFT" if x_use < 0 else "TURN RIGHT",
                    bbox=bbox_out,
                    conf=best_conf,
                    track_id=track_out,
                    x_dev_smooth=x_use,
                    area_ratio=area_ratio_out,
                )
            return FollowStepResult(
                cmd="S:0",
                status="TURN PAUSE",
                bbox=bbox_out,
                conf=best_conf,
                track_id=track_out,
                x_dev_smooth=x_use,
                area_ratio=area_ratio_out,
            )

        if area_ratio_out < cfg.stop_area_ratio:
            return FollowStepResult(
                cmd=f"F:{cfg.forward_speed}",
                status="FORWARD",
                bbox=bbox_out,
                conf=best_conf,
                track_id=track_out,
                x_dev_smooth=x_use,
                area_ratio=area_ratio_out,
            )
        return FollowStepResult(
            cmd="S:0",
            status="STOP (1m reached)",
            bbox=bbox_out,
            conf=best_conf,
            track_id=track_out,
            x_dev_smooth=x_use,
            area_ratio=area_ratio_out,
        )

    state.prev_track_id = None
    state.prev_center = None

    in_search_pulse = now < state.search_active_until

    if in_search_pulse and state.search_current_cmd:
        return FollowStepResult(
            cmd=state.search_current_cmd,
            status="SEARCHING (scan)",
            bbox=None,
            conf=0.0,
            track_id=None,
            x_dev_smooth=state.x_dev_smooth,
            area_ratio=0.0,
        )

    if not in_search_pulse:
        state.lost_frames += 1

    if state.lost_frames >= cfg.lost_frames_before_search:
        direction = "left" if (state.search_phase // cfg.search_alternate_every) % 2 == 0 else "right"
        speed = cfg.turn_speed_min
        state.search_current_cmd = f"L:{speed}" if direction == "left" else f"R:{speed}"
        state.search_active_until = now + cfg.search_turn_duration
        state.search_phase += 1
        state.lost_frames = 0
        return FollowStepResult(
            cmd=state.search_current_cmd,
            status="SEARCHING (scan)",
            bbox=None,
            conf=0.0,
            track_id=None,
            x_dev_smooth=state.x_dev_smooth,
            area_ratio=0.0,
        )

    return FollowStepResult(
        cmd="S:0",
        status="SEARCHING",
        bbox=None,
        conf=0.0,
        track_id=None,
        x_dev_smooth=state.x_dev_smooth,
        area_ratio=0.0,
    )


def follow_config_from_env() -> FollowConfig:
    import os

    return FollowConfig(
        target_class=os.getenv("TARGET_CLASS", "person"),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.4")),
        forward_speed=int(os.getenv("FORWARD_SPEED", "150")),
        turn_speed_min=int(os.getenv("TURN_SPEED_MIN", "115")),
        turn_speed_max=int(os.getenv("TURN_SPEED_MAX", "200")),
        center_tolerance=float(os.getenv("CENTER_TOLERANCE", "0.05")),
        stop_area_ratio=float(os.getenv("STOP_AREA_RATIO", "0.35")),
        lost_frames_before_search=int(os.getenv("LOST_FRAMES_BEFORE_SEARCH", "20")),
        search_turn_duration=float(os.getenv("SEARCH_TURN_DURATION", "0.25")),
        search_alternate_every=int(os.getenv("SEARCH_ALTERNATE_EVERY", "3")),
        smooth_alpha=float(os.getenv("X_DEV_SMOOTH_ALPHA", "0.35")),
        turn_pulse_sec=float(os.getenv("TURN_PULSE_SEC", "0")),
    )
