"""ComfyUI-EasyControl-KSamplerCompat.

A single model-patch node, ``AnimaEasyControlPatch`` (display: "Anima
EasyControl (KSampler)"), that applies Anima EasyControl reference-image
conditioning as a ModelPatcher patch and returns a MODEL for the stock KSampler.

Unlike the upstream Flux/Qwen EasyControl ComfyUI nodes — which require a
dedicated model loader and a custom sampler node — this node takes MODEL + VAE +
IMAGE and returns a MODEL, so the conditioning rides through the normal KSampler
path and composes with the rest of the graph.

Targets ComfyUI's native Anima/Cosmos DiT backbone (comfy/ldm/cosmos/predict2.py).
EasyControl checkpoints (networks.methods.easycontrol) go in models/loras/.

Also ships two CPU-only prompt helpers for **EasyEdit** — the ``subject_edit``
adapter used as a reference-image instruction editor. They build the
delta-caption prompt ("additions, -removals") that adapter was trained on; see
``delta_caption.py`` and the EasyEdit section of the README.
"""

from .delta_caption import NODE_CLASS_MAPPINGS as _DELTA_CLASSES
from .delta_caption import NODE_DISPLAY_NAME_MAPPINGS as _DELTA_NAMES
from .nodes import NODE_CLASS_MAPPINGS as _EC_CLASSES
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _EC_NAMES

NODE_CLASS_MAPPINGS = {**_EC_CLASSES, **_DELTA_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_EC_NAMES, **_DELTA_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
