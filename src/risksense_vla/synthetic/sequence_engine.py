"""Temporal sequence and annotation generation for synthetic hazard scenes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from risksense_vla.synthetic.scene_config import (
    HAZARD_TEMPLATES,
    SceneConfig,
    actions_for_hazard,
    objects_for_hazard,
)


@dataclass
class ObjectState:
    """Per-object tracking state within a sequence."""

    label: str
    track_id: str
    bbox_xyxy: list[int] = field(default_factory=lambda: [0, 0, 50, 50])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class AnnotatedFrame:
    """Per-frame annotation payload."""

    frame_idx: int
    objects: list[dict[str, Any]]
    hoi: dict[str, Any]
    hazard_score: dict[str, Any]
    occluded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotatedSequence:
    """A complete annotated scene sequence."""

    scene_id: str
    scene_config: SceneConfig
    frames: list[AnnotatedFrame]
    camera_angle: str = "front"


class SequenceEngine:
    """Generates temporally coherent HOI + hazard sequences."""

    def __init__(self, *, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)

    def generate(self, scene_config: SceneConfig, scene_id: str) -> list[AnnotatedSequence]:
        """Produce one AnnotatedSequence per camera angle."""
        sequences: list[AnnotatedSequence] = []
        for angle in scene_config.camera_angles:
            frames = self._build_frames(scene_config)
            frames = _apply_temporal_augmentations(frames, scene_config.occlusion_level)
            sequences.append(
                AnnotatedSequence(
                    scene_id=scene_id,
                    scene_config=scene_config,
                    frames=frames,
                    camera_angle=angle,
                )
            )
        return sequences

    def _build_frames(self, cfg: SceneConfig) -> list[AnnotatedFrame]:
        hazard_type = cfg.hazard_templates[0] if cfg.hazard_templates else "clutter"
        severity = HAZARD_TEMPLATES.get(hazard_type, "low")

        hazard_actions = actions_for_hazard(hazard_type)
        hazard_objects = objects_for_hazard(hazard_type)

        subject = "toddler" if hazard_type == "child_reach_danger" else "person"
        target_obj = random.choice(hazard_objects) if hazard_objects else "object"
        action_sequence = self._plan_action_sequence(hazard_actions, cfg.num_frames)

        obj_states = self._init_objects(cfg, subject, target_obj)

        frames: list[AnnotatedFrame] = []
        for t in range(cfg.num_frames):
            self._step_objects(obj_states, t, cfg)
            progress = t / max(1, cfg.num_frames - 1)
            action = action_sequence[t]
            score = self._hazard_curve(progress, severity)

            obj_dicts = [
                {"label": o.label, "bbox_xyxy": list(o.bbox_xyxy), "track_id": o.track_id}
                for o in obj_states
            ]

            frames.append(
                AnnotatedFrame(
                    frame_idx=t,
                    objects=obj_dicts,
                    hoi={
                        "subject": subject,
                        "action": action,
                        "object": target_obj,
                        "confidence": round(0.7 + 0.3 * progress, 3),
                    },
                    hazard_score={
                        "subject": subject,
                        "action": action,
                        "object": target_obj,
                        "score": round(score, 4),
                        "severity": severity,
                        "explanation": f"{subject} {action} {target_obj} ({hazard_type})",
                    },
                    metadata={"hazard_type": hazard_type, "progress": round(progress, 3)},
                )
            )
        return frames

    def _plan_action_sequence(self, actions: list[str], length: int) -> list[str]:
        """Create a plausible temporal ordering of actions across frames.

        Splits the sequence into phases, each dominated by one action, with
        smooth transitions between phases.
        """
        if not actions:
            return ["interact"] * length

        num_phases = min(len(actions), max(1, length // 6))
        chosen = random.sample(actions, num_phases) if num_phases <= len(actions) else actions
        phase_len = max(1, length // num_phases)

        seq: list[str] = []
        for i, act in enumerate(chosen):
            count = phase_len if i < num_phases - 1 else length - len(seq)
            seq.extend([act] * count)
        return seq[:length]

    def _init_objects(
        self, cfg: SceneConfig, subject: str, target_obj: str
    ) -> list[ObjectState]:
        w, h = cfg.resolution
        states: list[ObjectState] = []

        states.append(
            ObjectState(
                label=subject,
                track_id="t0",
                bbox_xyxy=[
                    random.randint(0, w // 4),
                    random.randint(h // 4, h // 2),
                    random.randint(w // 4, w // 2),
                    random.randint(h // 2, h),
                ],
                velocity=[random.uniform(1, 3), random.uniform(-1, 1)],
            )
        )

        states.append(
            ObjectState(
                label=target_obj,
                track_id="t1",
                bbox_xyxy=[
                    random.randint(w // 2, 3 * w // 4),
                    random.randint(h // 4, h // 2),
                    random.randint(3 * w // 4, w - 1),
                    random.randint(h // 2, 3 * h // 4),
                ],
                velocity=[0.0, 0.0],
            )
        )

        extra_pool = [
            o for o in cfg.object_classes if o not in {subject, target_obj}
        ]
        num_extra = random.randint(0, min(3, len(extra_pool)))
        for i, label in enumerate(random.sample(extra_pool, num_extra) if extra_pool else []):
            states.append(
                ObjectState(
                    label=label,
                    track_id=f"t{i + 2}",
                    bbox_xyxy=[
                        random.randint(0, w - 80),
                        random.randint(0, h - 80),
                        random.randint(40, w - 1),
                        random.randint(40, h - 1),
                    ],
                    velocity=[random.uniform(-1, 1), random.uniform(-1, 1)],
                )
            )

        return states

    def _step_objects(
        self, states: list[ObjectState], _t: int, cfg: SceneConfig
    ) -> None:
        """Move objects along smooth trajectories with jitter."""
        w, h = cfg.resolution
        for obj in states:
            jx = random.gauss(0, 1.5)
            jy = random.gauss(0, 1.5)
            dx = obj.velocity[0] + jx
            dy = obj.velocity[1] + jy
            obj.bbox_xyxy = [
                _clamp(int(obj.bbox_xyxy[0] + dx), 0, w - 40),
                _clamp(int(obj.bbox_xyxy[1] + dy), 0, h - 40),
                _clamp(int(obj.bbox_xyxy[2] + dx), 40, w - 1),
                _clamp(int(obj.bbox_xyxy[3] + dy), 40, h - 1),
            ]
            if obj.bbox_xyxy[2] <= obj.bbox_xyxy[0]:
                obj.bbox_xyxy[2] = obj.bbox_xyxy[0] + 40
            if obj.bbox_xyxy[3] <= obj.bbox_xyxy[1]:
                obj.bbox_xyxy[3] = obj.bbox_xyxy[1] + 40

    @staticmethod
    def _hazard_curve(progress: float, severity: str) -> float:
        """Monotonically increasing hazard score that ramps with progress."""
        base = {"high": 0.6, "medium": 0.3, "low": 0.1}.get(severity, 0.2)
        peak = {"high": 0.95, "medium": 0.7, "low": 0.45}.get(severity, 0.5)
        return base + (peak - base) * (1 - math.exp(-3 * progress))


def _apply_temporal_augmentations(
    frames: list[AnnotatedFrame], occlusion_prob: float
) -> list[AnnotatedFrame]:
    """Apply frame-skip and occlusion augmentations (ported from legacy script)."""
    frame_skip = random.choice([1, 1, 2, 3])
    out: list[AnnotatedFrame] = []
    for i, frame in enumerate(frames):
        if i % frame_skip != 0:
            continue
        if random.random() < occlusion_prob:
            frame = AnnotatedFrame(
                frame_idx=frame.frame_idx,
                objects=frame.objects,
                hoi=frame.hoi,
                hazard_score=frame.hazard_score,
                occluded=True,
                metadata=frame.metadata,
            )
        out.append(frame)
    if not out:
        out = frames[:1]
    return out


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))
