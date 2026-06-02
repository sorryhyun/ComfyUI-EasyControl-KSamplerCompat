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
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
