# ComfyUI-EasyControl-KSamplerCompat

KSampler-compatible **Anima EasyControl** image conditioning for ComfyUI.

One node — **`Anima EasyControl (KSampler)`** — takes `MODEL + VAE + IMAGE` and
returns a `MODEL` plus an empty `LATENT` sized to the conditioning resolution.
The reference-image conditioning rides on the returned model as a set of
ModelPatcher object-patches, so it flows straight into the **stock KSampler** and
composes with the rest of your graph (schedulers, ControlNet, other adapters).
Wire the `LATENT` output into the KSampler's `latent_image` and the generation
matches the (aspect-preserved, ~1MP) cond grid automatically — no manual
`EmptyLatentImage` sizing, and the two streams share a grid for clean spatial
alignment on tasks like colorization.

This is deliberately unlike the upstream Flux / Qwen EasyControl ComfyUI nodes,
which make you load a model through a dedicated loader and sample with a
**dedicated sampler node**. Here there is no custom sampler — just a model patch.

## What it's for

EasyControl is a frozen-DiT reference-image adapter for the Anima model. The
flagship use case is **colorization**: feed a grayscale / manga line page into
the `IMAGE` socket and a color prompt into your normal text encode, and the DiT
colorizes the reference while respecting its structure.

## Install

1. Clone into `ComfyUI/custom_nodes/`:
   ```
   git clone https://github.com/sorryhyun/ComfyUI-EasyControl-KSamplerCompat
   ```
2. Put your EasyControl checkpoint (`*.safetensors` trained with
   `networks.methods.easycontrol`, e.g. the colorize adapter) in
   `ComfyUI/models/loras/`.
3. Restart ComfyUI.

## Wiring

```
UNETLoader ──► MODEL ─┐                          MODEL ──► KSampler ──► ...
                      ├─► Anima EasyControl ────┤            ▲
VAELoader ───► VAE ───┤        (KSampler)        └─ LATENT ──┘
LoadImage ──► IMAGE ──┘            ▲
                          easycontrol_lora = <your ckpt>
                          strength = 1.0, target_megapixels = 1.0
```

- **`image`** — the conditioning image (the grayscale / lineart page for
  colorization). It is resized to at most `target_megapixels` (aspect-preserving,
  VAE/patch-snapped) and VAE-encoded once; the returned `LATENT` is sized to that
  same grid, so the cond stream and the sampled target share a resolution.
- **`strength`** — scales the trained cond effect. `1.0` = as trained. Raise to
  push harder toward the reference, lower to loosen.
- **`target_megapixels`** (optional) — max output size in megapixels for the
  returned latent and the cond encode. `1.0` (default) keeps generation on the
  ~1MP distribution Anima was trained at by downscaling anything larger; a source
  already below the cap is left at its native resolution (never upscaled). `0`
  keeps the input's native resolution (still snapped to the VAE/patch grid). Feed
  the `LATENT` output into the KSampler's `latent_image`.
- **`cond_scale_override`** (optional) — replace the checkpoint's trained
  `cond_scale` outright (before `strength`); `0` = keep the trained value.

Text prompt: encode it the normal way and feed the KSampler's positive/negative
as usual. For the colorize adapter, the prompt carries **color** facts the
lineart can't (`"pink hair, blue eyes, white dress"`); structure comes from the
reference image.

## How it works

EasyControl extends each DiT block's self-attention to attend over a
**reference-image key/value stream** in addition to the target tokens, gated by
a trained per-block scalar bias `b_cond`. On apply:

1. The reference image is VAE-encoded and mapped into the DiT input space
   (`process_latent_in`).
2. On the first sampling step, the cond stream is walked once through all blocks
   to build a per-block `(cond_k, cond_v)` cache (deterministic across steps —
   the cond t-embedding is fixed at `t=0`). Reused for every step and CFG
   branch.
3. Each block's `forward` is replaced (via reversible object-patch) with one
   that runs extended self-attention `[target_k ; cond_k]` with the `b_cond`
   bias on the cond columns. Cross-attention and MLP run baseline.

It targets ComfyUI's native Anima/Cosmos backbone
(`comfy/ldm/cosmos/predict2.py`) directly — a from-scratch reimplementation of
anima_lora's inference path against ComfyUI's **split** `q_proj/k_proj/v_proj`
layout. The trained cond-LoRA delta (a fused `D→3D` tensor) is sliced into q/k/v
thirds and added onto the split projections; the base weights are numerically
identical between the two DiTs (same pretrained model, fused-vs-split layout),
so this reproduces what training saw. No anima_lora vendoring required — it uses
only ComfyUI's own modules plus the checkpoint tensors.

## Known limitations / notes

- **Latent space.** The node assumes `VAE.encode` → `process_latent_in` yields
  the same latent the DiT receives for the noisy target. This is the principled
  match for the native Anima model; if a future build changes the latent
  pipeline, the cond stream would need the same change.
- **Block-compile ordering.** If you also use a node that *rebuilds* the DiT
  (e.g. Anima block-compile), apply **this node after** it in the chain. The
  per-block patches resolve their block from the live diffusion_model each
  forward (rebuild-tolerant), but as with the other Anima nodes, mixing
  DiT-rebuilding patches is best avoided.
- **Attention.** Extended attention uses `scaled_dot_product_attention` with an
  additive bias mask (memory-efficient backends handle the bias). This differs
  from anima_lora's training-time flash-LSE path numerically only at the ulp
  level.
- **One reference image.** If a batch is fed to `IMAGE`, the first image is used.

## Relationship to the Anima Adapter Loader

This is a separate repo from
[`ComfyUI-Anima_lora-Adapter`](https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter)
(LoRA / HydraLoRA / ReFT / FeRA / Soft Tokens). Those add residuals to whole
Linear/block outputs and can ride generic forward hooks; EasyControl reaches
inside self-attention, so it needs this dedicated reimplementation.
