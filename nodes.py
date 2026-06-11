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
import torch
import torch.nn.functional as F

from .easycontrol_patch import install_easycontrol


def _vae_spatial(vae):
    """Spatial downscale (h_mult, w_mult) of a ComfyUI VAE.

    ``downscale_ratio`` is either an int (square) or a ``(callable, h, w)`` tuple
    for the temporally-aware video VAEs. We only need the spatial multiples.
    """
    dr = getattr(vae, "downscale_ratio", 8)
    if isinstance(dr, (tuple, list)):
        return int(dr[1]), int(dr[2])
    return int(dr), int(dr)


def _resize_to_megapixels(image, vae, target_mp):
    """Resize an IMAGE ``[B,H,W,C]`` (in [0,1]) to at most ``target_mp``
    megapixels, aspect-preserving, snapping each side to ``vae_spatial *
    patch(2)`` so the resulting latent has even dims (cosmos ``patch_spatial=2``).

    ``target_mp`` is a MAX cap: a source already below it is left at its native
    resolution (only snapped) rather than upscaled.

    ``target_mp <= 0`` means keep the native resolution (still snapped, so the
    VAE/patch grid is clean). Returns the resized image ``[B,H,W,C]``.
    """
    _, h, w, _ = image.shape
    sh, sw = _vae_spatial(vae)
    mult_h, mult_w = sh * 2, sw * 2

    if target_mp and target_mp > 0:
        # target_mp is a MAX cap: only downscale when the source exceeds it,
        # never upscale a smaller source (clamp scale to <= 1.0).
        scale = min(1.0, (target_mp * 1_000_000 / float(h * w)) ** 0.5)
        new_h, new_w = h * scale, w * scale
    else:
        new_h, new_w = float(h), float(w)

    snap_h = max(mult_h, int(round(new_h / mult_h)) * mult_h)
    snap_w = max(mult_w, int(round(new_w / mult_w)) * mult_w)
    if (snap_h, snap_w) == (h, w):
        return image

    # IMAGE [B,H,W,C] → [B,C,H,W] for interpolate → back.
    chw = image.permute(0, 3, 1, 2)
    chw = F.interpolate(chw, size=(snap_h, snap_w), mode="bilinear", align_corners=False)
    return chw.permute(0, 2, 3, 1).contiguous()


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
                "target_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": (
                            "Output size, in megapixels, for the returned LATENT "
                            "(and the conditioning encode). The image is resized "
                            "aspect-preserving to ~this many MP and snapped to the "
                            "VAE/patch grid; the cond stream is encoded at the same "
                            "resolution so it shares the target's grid (best spatial "
                            "alignment for colorize). 1.0 keeps generation on the "
                            "~1MP distribution Anima was trained at. 0 = keep the "
                            "input's native resolution."
                        ),
                    },
                ),
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

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima EasyControl image conditioning, KSampler-compatible. Feeds a "
        "reference image (VAE-encoded) into a frozen-DiT extended self-attention "
        "stream and returns a patched MODEL plus an empty LATENT sized to the "
        "conditioning resolution — wire the latent straight into the KSampler so "
        "it generates at the (aspect-matched, ~1MP) cond grid. Ideal for spatially"
        " aligned tasks like colorization."
    )

    def apply(self, model, vae, image, base_model, easycontrol_lora, strength,
              target_megapixels=1.0, cond_scale_override=0.0):
        # base_model is an informational guard — only "anima" is offered today.
        del base_model
        new_model = model.clone()
        weight_path = folder_paths.get_full_path("loras", easycontrol_lora)
        if weight_path is None:
            raise FileNotFoundError(
                f"EasyControl checkpoint not found in loras folder: {easycontrol_lora}"
            )

        # IMAGE is [B, H, W, C] in [0, 1]; use the first image as the reference.
        # Resize to ~target_megapixels (aspect-preserving, VAE/patch-snapped) so
        # the cond encode and the returned target latent share one grid.
        pixels = image[:1, :, :, :3]
        pixels = _resize_to_megapixels(pixels, vae, float(target_megapixels))
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

        # Empty target latent matching the cond grid. KSampler overwrites the
        # contents with noise (full denoise); only the shape matters here.
        out_latent = {"samples": torch.zeros_like(vae_latent)}
        return (new_model, out_latent)


NODE_CLASS_MAPPINGS = {
    "AnimaEasyControlPatch": AnimaEasyControlPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaEasyControlPatch": "Anima EasyControl (KSampler)",
}
