"""EasyControl image-conditioning node for the Anima DiT — KSampler-compatible.

A single model-patch node: ``MODEL + VAE + IMAGE -> MODEL``. The returned MODEL
carries the reference-image conditioning as a set of object patches, so it flows
straight into the stock KSampler (and composes with ControlNet, schedulers,
other adapters) — no dedicated sampler node, unlike the upstream Flux/Qwen
EasyControl ComfyUI nodes.

The EasyControl ``.safetensors`` goes in ``ComfyUI/models/loras/`` (same folder
the LoRA loaders read). Feed the conditioning image (e.g. the grayscale / manga
line page for colorization) into the IMAGE socket and a VAE that matches the
Anima latent space into the VAE socket.
"""

import folder_paths

from .easycontrol_patch import install_easycontrol


class AnimaEasyControlPatch:
    """Apply Anima EasyControl reference-image conditioning to a MODEL.

    EasyControl is a frozen-DiT adapter: a per-block cond LoRA on self-attn
    (q/k/v/o) + FFN plus a scalar ``b_cond`` logit-bias gate. At inference the
    reference image is VAE-encoded, the cond stream is walked once to build a
    per-block key/value cache, and each block's self-attention is extended to
    attend over those cached cond keys. Output is a MODEL for the stock KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "base_model": (
                    ["anima"],
                    {
                        "default": "anima",
                        "tooltip": (
                            "Base model family this node supports. Currently "
                            "Anima only — the EasyControl reimplementation targets "
                            "the Anima/Cosmos DiT backbone. More families may be "
                            "added later."
                        ),
                    },
                ),
                "easycontrol_lora": (
                    loras,
                    {
                        "tooltip": (
                            "EasyControl checkpoint (networks.methods.easycontrol "
                            "— per-block cond LoRA + b_cond gate). Lives in "
                            "ComfyUI/models/loras/."
                        )
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Conditioning strength. Scales the cond-LoRA deltas "
                            "(multiplies the trained cond_scale). 1.0 = as trained; "
                            "raise to push harder toward the reference, lower to "
                            "loosen."
                        ),
                    },
                ),
            },
            "optional": {
                "cond_scale_override": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "If > 0, replaces the checkpoint's trained cond_scale "
                            "(before strength). 0 = use the trained value."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima EasyControl image conditioning, KSampler-compatible. Feeds a "
        "reference image (VAE-encoded) into a frozen-DiT extended self-attention "
        "stream and returns a patched MODEL for the stock KSampler."
    )

    def apply(self, model, vae, image, base_model, easycontrol_lora, strength,
              cond_scale_override=0.0):
        # base_model is an informational guard — only "anima" is offered today.
        del base_model
        new_model = model.clone()
        weight_path = folder_paths.get_full_path("loras", easycontrol_lora)
        if weight_path is None:
            raise FileNotFoundError(
                f"EasyControl checkpoint not found in loras folder: {easycontrol_lora}"
            )

        # IMAGE is [B, H, W, C] in [0, 1]; use the first image as the reference.
        pixels = image[:1, :, :, :3]
        vae_latent = vae.encode(pixels)  # ComfyUI latent space [1, C, h, w]

        # Map into the DiT's input space — the same transform apply_model applies
        # to the noisy latent before calling diffusion_model. Without this the
        # cond stream sees a differently-scaled latent than the target stream.
        cond_latent = new_model.model.process_latent_in(vae_latent)

        override = float(cond_scale_override) if cond_scale_override and cond_scale_override > 0 else None
        install_easycontrol(
            new_model, weight_path, cond_latent, float(strength),
            cond_scale_override=override,
        )
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaEasyControlPatch": AnimaEasyControlPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaEasyControlPatch": "Anima EasyControl (KSampler)",
}
