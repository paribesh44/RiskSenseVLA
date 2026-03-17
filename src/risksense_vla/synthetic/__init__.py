"""Synthetic hazard dataset generation pipeline."""

from risksense_vla.synthetic.io_export import DatasetWriter
from risksense_vla.synthetic.renderers import (
    ProceduralRenderer,
    RendererProtocol,
    StableDiffusionRenderer,
    get_renderer,
)
from risksense_vla.synthetic.scene_config import (
    DEFAULT_ACTION_TEMPLATES,
    DEFAULT_OBJECT_CLASSES,
    HAZARD_TEMPLATES,
    ROOM_PRESETS,
    SceneConfig,
    build_scene_configs,
)
from risksense_vla.synthetic.sequence_engine import (
    AnnotatedFrame,
    AnnotatedSequence,
    SequenceEngine,
)

__all__ = [
    "AnnotatedFrame",
    "AnnotatedSequence",
    "DatasetWriter",
    "DEFAULT_ACTION_TEMPLATES",
    "DEFAULT_OBJECT_CLASSES",
    "HAZARD_TEMPLATES",
    "ProceduralRenderer",
    "RendererProtocol",
    "ROOM_PRESETS",
    "SceneConfig",
    "SequenceEngine",
    "StableDiffusionRenderer",
    "build_scene_configs",
    "get_renderer",
]
