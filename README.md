# ComfyUI-EasyControl-KSamplerCompat

KSampler-compatible **Anima EasyControl** image conditioning for ComfyUI.

The main node — **`Anima EasyControl (KSampler)`** — takes `MODEL + VAE + IMAGE`
and returns a `MODEL` plus an empty `LATENT` sized to the conditioning
resolution.
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

Two CPU-only helper nodes ride along for **EasyEdit**, the instruction-editing
use of the same path — see [EasyEdit](#easyedit--edit-by-instruction-no-inversion).

## What it's for

EasyControl is a frozen-DiT reference-image adapter for the Anima model. The
flagship use case is **colorization**: feed a grayscale / manga line page into
the `IMAGE` socket and a color prompt into your normal text encode, and the DiT
colorizes the reference while respecting its structure.

The second use is **EasyEdit**: with the `subject_edit` checkpoint the same node
becomes a reference + instruction editor — give it a character and a list of
changes, get that character re-rendered with the changes applied, in one pass
and with no inversion.

## Install

1. Clone into `ComfyUI/custom_nodes/`:
   ```
   git clone https://github.com/sorryhyun/ComfyUI-EasyControl-KSamplerCompat
   ```
2. Put your EasyControl checkpoint (`*.safetensors` trained with
   `networks.methods.easycontrol`) in `ComfyUI/models/loras/`. Published
   adapters live in
   [`sorryhyun/anima-easycontrol-adapters`](https://huggingface.co/sorryhyun/anima-easycontrol-adapters):
   `anima_colorize_v2.safetensors` (colorization) and
   `anima_subject_edit_alpha.safetensors` (EasyEdit).
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
- **`mask`** (optional) — an inpaint mask for the `inpaint` adapter. Wire the
  **original** image into `image` and a mask here; the node fills the white
  (selected) region with mid-gray and encodes that as the conditioning image —
  reproducing what the `inpaint` adapter trained on (cond = original image with a
  free-form gray hole). The DiT then regenerates the hole consistent with the
  surrounding pixels and the prompt. Leave unconnected for adapters that take a
  pre-built cond image (e.g. colorize).
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

## EasyEdit — edit by instruction, no inversion

**EasyEdit is a checkpoint plus a prompting convention, not a new node.** It
rides the exact path above: reference image into `image`, an *edit instruction*
into your positive text encode, one plain KSampler run. No inversion pass, no
anchor, no mask, no offset tuning.

- **Checkpoint**: `anima_subject_edit_alpha.safetensors` from
  [`sorryhyun/anima-easycontrol-adapters`](https://huggingface.co/sorryhyun/anima-easycontrol-adapters)
  → `ComfyUI/models/loras/`.
- **Workflow**: [`workflows/easyedit.json`](workflows/easyedit.json) — drag it
  into ComfyUI. Core nodes plus this repo only.

### What it does

Give it a picture of a character and a list of changes; it re-renders that
character with the changes applied. It preserves **identity and appearance**,
not composition — expect a new pose/framing that is recognizably the same
character. That is the opposite trade from the
[DirectEdit node](https://github.com/sorryhyun/anima_lora), which preserves the
original pixels and edits in place at the cost of a second (inversion) pass:

| | DirectEdit | EasyEdit |
|---|---|---|
| Semantics | edit THIS image in place | re-render with these changes |
| Preserves | composition + unchanged pixels | identity / appearance |
| Cost | 2 passes | 1 generation |
| Inputs | image + source/target captions | image + delta instruction |
| Weak spot | hard images | object removals |

### The prompting convention

The prompt is a **tag delta**, not a caption. Additions first as bare tags, then
removals prefixed with `-`:

```
wading in water, night, smile, -chair, -ramune
```

Rules that matter:

- **Instruction, never a full caption.** Only what *changes* goes in the prompt.
  Tags the reference already satisfies must be left out — including the
  character's name. Identity comes from the image, not the text; naming the
  character in the prompt is the failure mode the training objective was
  designed to starve.
- **Anima tag vocabulary**, comma-separated (`blue eyes`, not `blue_eyes` — the
  builder nodes normalize this for you).
- **The rating band participates.** `safe` / `sensitive` / `nsfw` / `explicit`
  are ordinary tags, so `nsfw, -safe` converts the rating. That is a real
  capability, and it means an accidental rating tag in your instruction will
  move the output.
- **Do not add quality tags** (`masterpiece`, `best quality`) to the
  instruction. Training saw deltas only; anything else is off-convention.
  Quality *negatives* on the negative prompt are fine and are what the workflow
  ships with.

### Building the instruction

Two CPU-only helper nodes ship here so you don't hand-compose the string:

- **`Anima EasyEdit Instruction (add / remove)`** — two text fields, one
  correctly formatted instruction out. Use this when you know what you want to
  change.
- **`Anima EasyEdit Delta (caption diff)`** — two full captions in
  (`source_caption`, `target_caption`), the delta out. Wire the
  [Anima Tagger](https://github.com/sorryhyun/anima_lora) node's caption output
  into `source_caption`, paste an edited copy into `target_caption`, and you get
  "edit the image by editing its caption" — the DirectEdit UX without the
  inversion pass. Shared tags cancel automatically, so you cannot accidentally
  leave the character name in. It also outputs the computed `additions` and
  `removals` separately, which is the quickest way to see why an edit did
  something you didn't ask for.

Both emit the same format the training-time pair miner produced (there is a test
pinning that equivalence), and both accept comma- **or** newline-separated tags.

### Settings

Defaults are the validated operating point — the workflow ships with them:

| | |
|---|---|
| `strength` | `1.0` |
| `target_megapixels` | `1.0` |
| `cond_scale_override` | `0` (use the trained value) |
| sampler / scheduler | `euler` / `simple` |
| steps / cfg | `28` / `4.0` |

ComfyUI's built-in Anima config already uses `shift = 3.0`, which is the shift
the adapter was validated at, so no extra sampling-shift node is needed.

**There is deliberately no `b_cond` offset knob.** This node applies the
checkpoint's trained gate verbatim — equivalent to `b_offset 0` in the
anima_lora CLI — and that *is* EasyEdit's operating point. The engaged band is
narrow and centered there: raising the gate by +2 pulls the output back toward
the reference's composition and +3 is a near-verbatim copy of the reference.
If an edit isn't landing, change the instruction, not the gate.

### Limits (alpha)

The checkpoint is published as `_alpha` for these reasons; know them before you
file a bug:

- **Removals of objects that are physically present in the reference mostly
  fail.** `-ramune`, `-inflatable orca`, `-hair beads` largely survive the
  instruction. Additions and *state changes* ("jacket partially removed") are
  where the adapter is strong. Negation is adapter-side only — the text encoder
  reads `-ramune` as "ramune", so a weak adapter response leaves you with an
  attractor, not a removal. If a removal matters, phrase the change as an
  addition where you can.
- **Validation is narrow.** The published evidence is a render-judged
  instruction probe on three characters at a single seed, drawn from the
  training pair set — an upper bound, not a held-out score. Off-corpus images
  (non-anime, or styles the base model covers thinly) are untested.
- **Composition is not preserved by design.** If you need the original framing
  and background pixels intact, use DirectEdit, not this.

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

Checkpoints trained with **`train_adaln`** (target-stream AdaLN LoRA — per-block
deltas on the `adaln_modulation_{self_attn,cross_attn,mlp}` up-projections) are
fully supported: the deltas are merged as ordinary ModelPatcher weight patches
(scaled by `strength`, composing with lowvram loading, block compile, and other
LoRAs), while the cond-stream prefill subtracts them again so the reference
stream sees the frozen modulation exactly as training did. Channel-scaled
checkpoints (`*.inv_scale` tensors from anima_lora's per-channel gradient
rebalance) are also applied faithfully. The loader is **strict**: a checkpoint
carrying tensors this node doesn't implement is refused with an error instead of
silently dropping them (a dropped trained feature degrades output with no
warning — if you hit this, update the node).

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
