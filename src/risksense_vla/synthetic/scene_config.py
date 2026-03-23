"""Scene configuration, templates, and factory for synthetic hazard generation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SceneConfig:
    """Full specification for a single synthetic scene."""

    room_type: str = "kitchen"
    lighting: str = "bright"
    occlusion_level: float = 0.1
    object_classes: list[str] = field(default_factory=list)
    action_templates: list[str] = field(default_factory=list)
    hazard_templates: list[str] = field(default_factory=list)
    num_frames: int = 24
    resolution: tuple[int, int] = (640, 480)
    camera_angles: list[str] = field(default_factory=lambda: ["front"])
    fps: int = 24


ROOM_PRESETS: dict[str, dict[str, Any]] = {
    "kitchen": {
        "typical_objects": ["person", "knife", "stove", "hot_pan", "glass", "bottle"],
        "bg_color": (200, 220, 210),
        "lighting_options": ["bright", "dim", "overhead"],
    },
    "living_room": {
        "typical_objects": ["person", "glass", "bottle", "electrical_cord", "toddler"],
        "bg_color": (210, 200, 190),
        "lighting_options": ["bright", "dim", "natural"],
    },
    "workshop": {
        "typical_objects": ["person", "knife", "sharp_scissors", "chemical_bottle", "vehicle"],
        "bg_color": (180, 180, 180),
        "lighting_options": ["bright", "fluorescent", "dim"],
    },
    "nursery": {
        "typical_objects": ["toddler", "person", "bottle", "electrical_cord", "stove"],
        "bg_color": (220, 210, 230),
        "lighting_options": ["bright", "natural", "nightlight"],
    },
}

DEFAULT_OBJECT_CLASSES: list[str] = [
    "person",
    "knife",
    "stove",
    "vehicle",
    "glass",
    "bottle",
    "toddler",
    "hot_pan",
    "chemical_bottle",
    "electrical_cord",
    "wet_floor",
    "sharp_scissors",
]

DEFAULT_ACTION_TEMPLATES: list[str] = [
    "hold",
    "cut",
    "pour",
    "open",
    "touch_hot_surface",
    "carry",
    "drop",
    "close",
    "push",
    "pull",
    "move",
    "inspect",
    "crawl_toward",
    "reach_for",
    "spill",
    "knock_over",
    "slip_on",
]

HAZARD_TEMPLATES: dict[str, str] = {
    "hot_surface_contact": "high",
    "sharp_tool_contact": "high",
    "spill_risk": "medium",
    "trip_obstacle": "medium",
    "clutter": "low",
    "child_reach_danger": "high",
    "electrical_contact": "high",
    "chemical_exposure": "high",
    "crowd_crush": "medium",
    "occlusion_hidden_hazard": "medium",
}

RARE_HAZARD_TYPES: set[str] = {
    "child_reach_danger",
    "electrical_contact",
    "chemical_exposure",
    "crowd_crush",
    "occlusion_hidden_hazard",
}

_HAZARD_TO_ACTIONS: dict[str, list[str]] = {
    "hot_surface_contact": ["touch_hot_surface", "reach_for"],
    "sharp_tool_contact": ["hold", "cut", "carry"],
    "spill_risk": ["pour", "spill", "knock_over"],
    "trip_obstacle": ["move", "push", "slip_on"],
    "clutter": ["move", "push", "drop"],
    "child_reach_danger": ["crawl_toward", "reach_for"],
    "electrical_contact": ["reach_for", "hold"],
    "chemical_exposure": ["pour", "open", "spill"],
    "crowd_crush": ["push", "move"],
    "occlusion_hidden_hazard": ["move", "inspect"],
}

_HAZARD_TO_OBJECTS: dict[str, list[str]] = {
    "hot_surface_contact": ["stove", "hot_pan"],
    "sharp_tool_contact": ["knife", "sharp_scissors"],
    "spill_risk": ["glass", "bottle", "chemical_bottle"],
    "trip_obstacle": ["wet_floor", "electrical_cord"],
    "clutter": ["bottle", "glass", "knife"],
    "child_reach_danger": ["stove", "knife", "chemical_bottle"],
    "electrical_contact": ["electrical_cord"],
    "chemical_exposure": ["chemical_bottle"],
    "crowd_crush": ["person"],
    "occlusion_hidden_hazard": ["knife", "chemical_bottle", "hot_pan"],
}


def actions_for_hazard(hazard_type: str) -> list[str]:
    """Return plausible actions for a given hazard type."""
    return _HAZARD_TO_ACTIONS.get(hazard_type, ["interact"])


def objects_for_hazard(hazard_type: str) -> list[str]:
    """Return plausible objects for a given hazard type."""
    return _HAZARD_TO_OBJECTS.get(hazard_type, ["object"])


def build_scene_configs(
    *,
    num_scenes: int = 100,
    rare_event_ratio: float = 0.3,
    resolution: tuple[int, int] = (640, 480),
    enable_multi_angle: bool = False,
    num_frames: int = 24,
    fps: int = 24,
    config_yaml: str | None = None,
    seed: int | None = None,
) -> list[SceneConfig]:
    """Build a list of scene configs covering all hazard types.

    Ensures rare events appear at the requested ratio and every hazard type is
    represented at least once.
    """
    if seed is not None:
        random.seed(seed)

    if config_yaml is not None:
        return _load_from_yaml(config_yaml)

    all_hazards = list(HAZARD_TEMPLATES.keys())
    rare_hazards = [h for h in all_hazards if h in RARE_HAZARD_TYPES]
    common_hazards = [h for h in all_hazards if h not in RARE_HAZARD_TYPES]

    num_rare = max(len(rare_hazards), int(num_scenes * rare_event_ratio))

    hazard_sequence: list[str] = []
    for h in all_hazards:
        hazard_sequence.append(h)

    while len(hazard_sequence) < num_scenes:
        if len([h for h in hazard_sequence if h in RARE_HAZARD_TYPES]) < num_rare:
            hazard_sequence.append(random.choice(rare_hazards))
        else:
            hazard_sequence.append(random.choice(common_hazards))

    hazard_sequence = hazard_sequence[:num_scenes]
    random.shuffle(hazard_sequence)

    room_types = list(ROOM_PRESETS.keys())
    configs: list[SceneConfig] = []

    for hazard_type in hazard_sequence:
        room = random.choice(room_types)
        preset = ROOM_PRESETS[room]
        lighting = random.choice(preset["lighting_options"])
        occlusion = random.uniform(0.05, 0.25)
        angles = ["front"]
        if enable_multi_angle:
            angles = ["front", "left", "right"]

        obj_classes = list(set(preset["typical_objects"]) | set(objects_for_hazard(hazard_type)))
        act_templates = actions_for_hazard(hazard_type)

        configs.append(
            SceneConfig(
                room_type=room,
                lighting=lighting,
                occlusion_level=occlusion,
                object_classes=obj_classes,
                action_templates=act_templates,
                hazard_templates=[hazard_type],
                num_frames=num_frames,
                resolution=resolution,
                camera_angles=angles,
                fps=fps,
            )
        )

    return configs


def _load_from_yaml(path: str) -> list[SceneConfig]:
    """Load scene configs from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenes = raw if isinstance(raw, list) else raw.get("scenes", [raw])
    configs: list[SceneConfig] = []
    for entry in scenes:
        res = entry.get("resolution", [640, 480])
        configs.append(
            SceneConfig(
                room_type=entry.get("room_type", "kitchen"),
                lighting=entry.get("lighting", "bright"),
                occlusion_level=entry.get("occlusion_level", 0.1),
                object_classes=entry.get("object_classes", list(DEFAULT_OBJECT_CLASSES)),
                action_templates=entry.get("action_templates", list(DEFAULT_ACTION_TEMPLATES)),
                hazard_templates=entry.get("hazard_templates", list(HAZARD_TEMPLATES.keys())),
                num_frames=entry.get("num_frames", 24),
                resolution=(int(res[0]), int(res[1])),
                camera_angles=entry.get("camera_angles", ["front"]),
                fps=entry.get("fps", 24),
            )
        )
    return configs
