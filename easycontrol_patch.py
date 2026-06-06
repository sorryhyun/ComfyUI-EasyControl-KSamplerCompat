"""EasyControl image conditioning for ComfyUI — KSampler-compatible.

This is a from-scratch reimplementation of anima_lora's EasyControl *inference*
path, rewritten against ComfyUI's native Anima/Cosmos DiT backbone
(``comfy/ldm/cosmos/predict2.py``) so it runs under the stock KSampler with no
dedicated sampler node.

Why a reimplementation rather than vendoring anima_lora's network module:
EasyControl reaches *inside* each DiT block's self-attention (it concatenates a
reference-image key/value stream into the target's attention). anima_lora's
training DiT uses a **fused** ``self_attn.qkv_proj`` (D->3D) and an
``adaln_fused_down`` / ``adaln_up_*`` AdaLN-LoRA topology; ComfyUI's backbone
uses **split** ``q_proj`` / ``k_proj`` / ``v_proj`` and ``adaln_modulation_*``
Sequentials. So the in-block splice cannot be reused as-is.

Correctness hinge: the trained cond-LoRA delta is a fused ``D->3D`` tensor whose
three chunks are (q, k, v) in that order — exactly the split layout ComfyUI
uses. The *base* projection weights are numerically identical between the two
DiTs (same pretrained Anima model, just fused-vs-split layout — see anima_lora
``networks/attn_fuse.py``), so adding the sliced fused delta onto ComfyUI's
split q/k/v projections reproduces what training saw.

Mechanism, per denoising step:
  - The reference image is VAE-encoded once (node side) and the cond stream is
    walked once through all blocks (lazily, on the first DiT forward) to produce
    a per-block (cond_k, cond_v) cache. The cond stream is deterministic across
    steps (cond t-embedding at t=0, frozen weights), so this is computed once
    and reused for every step and every CFG branch.
  - Each block's self-attention is replaced by an *extended* attention over
    ``[target_k ; cond_k]`` / ``[target_v ; cond_v]`` with a per-block additive
    logit bias ``b_cond`` on the cond columns (the trained gate). Cross-attn and
    MLP run baseline. This extended attention runs on flash-attn (via an exact
    LSE decomposition of the joint softmax) when ComfyUI is launched with
    ``--use-flash-attention`` and ``flash_attn`` is installed; otherwise it uses
    a portable float-mask SDPA path. See ``_extended_self_attn``.

Robustness: the per-block forward closures resolve their block from
``state.live_blocks`` (refreshed by the pre-hook from the live diffusion_model
each forward) rather than capturing a block reference, so a downstream node that
rebuilds the DiT (e.g. block-compile) doesn't strand the patch on a dead
instance — same hazard the in-tree Anima nodes document.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# ComfyUI backbone helpers. Imported at module load — this node only makes sense
# inside a ComfyUI install, where these are always importable.
#
# Cosmos/Anima rope application. ComfyUI shipped the standalone
# ``apply_rotary_pos_emb(t, freqs)`` helper from June 2025 until #14181
# ("Speed up anima a bit on nvidia", in 0.24.0), which deleted the function and
# inlined its two call sites as ``comfy.quant_ops.ck.apply_rope_split_half(q, k,
# rope_emb)``. The ``rope_emb`` *format* is unchanged across that refactor — only
# the helper moved — so we keep a vendored copy of the original (numerically
# identical, split-half convention) and prefer the in-tree symbol when present.
try:  # ComfyUI < 0.24
    from comfy.ldm.cosmos.predict2 import apply_rotary_pos_emb as _comfy_apply_rope
except ImportError:  # ComfyUI >= 0.24 — standalone helper removed
    _comfy_apply_rope = None

# Optional flash-attention fast path for the extended self-attention. Mirrors
# ComfyUI's own gating (comfy/ldm/modules/attention.py): the kernel is used only
# when (a) ``flash_attn`` imports and (b) the user launched with
# ``--use-flash-attention`` (surfaced as ``model_management.flash_attention_enabled()``).
# Otherwise we fall back to the float-mask SDPA path, which works everywhere.
try:
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None

_FLASH_ENABLED_CACHE: Optional[bool] = None


def _flash_ext_enabled() -> bool:
    """True iff flash-attn is importable AND ComfyUI was launched with
    ``--use-flash-attention``. Result cached — the launch flag can't change
    mid-process."""
    global _FLASH_ENABLED_CACHE
    if _FLASH_ENABLED_CACHE is None:
        if _flash_attn_func is None:
            _FLASH_ENABLED_CACHE = False
        else:
            try:
                from comfy import model_management

                _FLASH_ENABLED_CACHE = bool(model_management.flash_attention_enabled())
            except Exception:
                _FLASH_ENABLED_CACHE = False
        if _FLASH_ENABLED_CACHE:
            logger.info("EasyControl: flash-attention extended-attn path enabled.")
    return _FLASH_ENABLED_CACHE


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply cosmos split-half RoPE to a single [B, S, H, D] tensor.

    Vendored from ComfyUI's pre-0.24 ``comfy.ldm.cosmos.predict2`` so the node
    survives the helper's removal. The newer in-tree path applies rope to q and
    k jointly (``apply_rope_split_half``); this per-tensor form is what the
    extended-attention splice needs (cond q/k are produced separately from the
    target stream), and it consumes the same ``freqs`` layout.
    """
    if _comfy_apply_rope is not None:
        return _comfy_apply_rope(t, freqs)
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_weights_and_meta(path: str):
    from safetensors import safe_open
    from safetensors.torch import load_file

    sd = load_file(path)
    meta = {}
    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
    return sd, meta


class _LoRA:
    """A single ``D->r->out`` cond-LoRA delta with an fp32 bottleneck.

    Mirrors anima_lora ``networks/methods/easycontrol.py::_LoRAProj``: the
    bottleneck runs in fp32 for bf16 stability, the ``alpha/r`` scale is folded
    in here, and the caller-side ``cond_scale * strength`` is applied via
    ``eff_scale`` at the call site.
    """

    __slots__ = ("down", "up", "scale")

    def __init__(self, down: torch.Tensor, up: torch.Tensor, alpha_over_r: float):
        self.down = down
        self.up = up
        self.scale = alpha_over_r

    def to(self, device, dtype):
        # Keep master copies in fp32 (the forward casts to fp32 anyway).
        self.down = self.down.to(device=device, dtype=torch.float32)
        self.up = self.up.to(device=device, dtype=torch.float32)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x.float(), self.down)
        h = F.linear(h, self.up)
        return (h * self.scale).to(x.dtype)


class EasyControlState:
    """Holds the trained cond-LoRA tensors, the reference latent, and the lazily
    built per-block KV cache. One per applied node instance; carried on the
    ModelPatcher via a strong ref so it isn't GC'd."""

    def __init__(self, sd: dict, meta: dict, strength: float,
                 cond_scale_override: Optional[float] = None):
        self.num_blocks = int(meta.get("ss_num_blocks", 28))
        self.r = int(meta.get("ss_cond_lora_dim", 16))
        self.alpha = float(meta.get("ss_cond_lora_alpha", float(self.r)))
        self.apply_ffn = bool(int(meta.get("ss_apply_ffn_lora", 1)))
        cond_scale = (
            float(cond_scale_override)
            if cond_scale_override is not None
            else float(meta.get("ss_cond_scale", 1.0))
        )
        # alpha/r folds into each _LoRA; cond_scale * strength is the call-site
        # effective scale (matches anima_lora get_effective_scale + multiplier).
        a_over_r = self.alpha / self.r if self.r > 0 else 1.0
        self.eff_scale = cond_scale * float(strength)

        def grab(prefix, idx):
            d = sd.get(f"{prefix}.{idx}.lora_down.weight")
            u = sd.get(f"{prefix}.{idx}.lora_up.weight")
            if d is None or u is None:
                return None
            return _LoRA(d, u, a_over_r)

        self.lora_qkv = [grab("cond_lora_qkv", i) for i in range(self.num_blocks)]
        self.lora_o = [grab("cond_lora_o", i) for i in range(self.num_blocks)]
        if self.apply_ffn:
            self.lora_ffn1 = [grab("cond_lora_ffn1", i) for i in range(self.num_blocks)]
            self.lora_ffn2 = [grab("cond_lora_ffn2", i) for i in range(self.num_blocks)]
        else:
            self.lora_ffn1 = [None] * self.num_blocks
            self.lora_ffn2 = [None] * self.num_blocks

        # b_cond stored as a ParameterList → keys "b_cond.{i}" (0-d tensors).
        self.b_cond = [
            sd.get(f"b_cond.{i}", torch.tensor(-10.0)).float().reshape(())
            for i in range(self.num_blocks)
        ]

        if any(q is None for q in self.lora_qkv):
            missing = [i for i, q in enumerate(self.lora_qkv) if q is None]
            raise ValueError(
                f"EasyControl checkpoint missing cond_lora_qkv for blocks {missing}. "
                "Is this an EasyControl (networks.methods.easycontrol) checkpoint?"
            )

        # Reference latent in *model space* (process_latent_in already applied by
        # the node). 4D [1, C, H, W]; given a device/dtype at prefill time.
        self.cond_latent: Optional[torch.Tensor] = None

        # Built lazily on first DiT forward.
        self.cond_kv_cache: Optional[list] = None
        self._cache_device = None
        # Refreshed each forward by the pre-hook from the live diffusion_model.
        self.live_blocks = None

    def _to(self, device, dtype):
        for lst in (self.lora_qkv, self.lora_o, self.lora_ffn1, self.lora_ffn2):
            for m in lst:
                if m is not None:
                    m.to(device, dtype)
        self.b_cond = [b.to(device=device) for b in self.b_cond]


# ---------------------------------------------------------------------------
# Cond stream prefill (build per-block KV cache)
# ---------------------------------------------------------------------------


def _cond_qkv(attn, cond_normed, lora_qkv, eff_scale, rope):
    """Split-projection cond Q/K/V with the fused LoRA delta sliced into thirds,
    then per-head norm + RoPE. Returns q, k, v as [B, S, H, D]."""
    q = attn.q_proj(cond_normed)
    k = attn.k_proj(cond_normed)
    v = attn.v_proj(cond_normed)
    if lora_qkv is not None:
        dq, dk, dv = (eff_scale * lora_qkv(cond_normed)).chunk(3, dim=-1)
        q = q + dq
        k = k + dk
        v = v + dv
    h, d = attn.n_heads, attn.head_dim
    q = rearrange(q, "b s (h d) -> b s h d", h=h, d=d)
    k = rearrange(k, "b s (h d) -> b s h d", h=h, d=d)
    v = rearrange(v, "b s (h d) -> b s h d", h=h, d=d)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    v = attn.v_norm(v)
    if rope is not None:
        q = apply_rotary_pos_emb(q, rope)
        k = apply_rotary_pos_emb(k, rope)
    return q, k, v


def _self_attn_bshd(q, k, v):
    """Plain self-attention on [B, S, H, D] tensors → [B, S, H, D]."""
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,S,D]
    out = F.scaled_dot_product_attention(qt, kt, vt)
    return out.transpose(1, 2)


@torch.no_grad()
def _build_cond_kv(dit, state: EasyControlState, device, dtype):
    """Walk the cond stream once; cache per-block (cond_k, cond_v).

    The cond stream is self-attention only (cross-attn is dropped on the cond
    side, matching anima_lora's two-stream variant), evolving cond_x block by
    block exactly as anima_lora ``precompute_cond_kv`` does, but against
    ComfyUI's split-projection block API.
    """
    state._to(device, dtype)
    cond_latent = state.cond_latent.to(device=device, dtype=dtype)
    if cond_latent.ndim == 4:
        cond_latent = cond_latent.unsqueeze(2)  # [B,C,1,H,W]

    cond_x5d, cond_rope_raw, cond_extra = dit.prepare_embedded_sequence(
        cond_latent, fps=None, padding_mask=None
    )
    cond_x = rearrange(cond_x5d, "b t h w d -> b (t h w) d")
    if cond_extra is not None:
        # Anima ships extra_per_block_abs_pos_emb=False, so this is None. If a
        # future config enables it, the per-block add would need mirroring here.
        logger.warning(
            "EasyControl: extra_per_block_abs_pos_emb is set; cond stream does "
            "not replicate it. Results may drift."
        )
    cond_rope = cond_rope_raw.unsqueeze(1).unsqueeze(0) if cond_rope_raw is not None else None

    # cond t-embedding at t=0 — identical to MiniTrainDIT._forward's t path.
    B = cond_x.shape[0]
    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    cemb, cadaln = dit.t_embedder[1](dit.t_embedder[0](zeros).to(dtype))
    cemb = dit.t_embedding_norm(cemb)

    eff = state.eff_scale
    cache = []
    for idx in range(state.num_blocks):
        block = dit.blocks[idx]
        attn = block.self_attn

        if block.use_adaln_lora:
            s_sa, sc_sa, g_sa = (
                block.adaln_modulation_self_attn(cemb) + cadaln
            ).chunk(3, dim=-1)
            s_mlp, sc_mlp, g_mlp = (
                block.adaln_modulation_mlp(cemb) + cadaln
            ).chunk(3, dim=-1)
        else:
            s_sa, sc_sa, g_sa = block.adaln_modulation_self_attn(cemb).chunk(3, dim=-1)
            s_mlp, sc_mlp, g_mlp = block.adaln_modulation_mlp(cemb).chunk(3, dim=-1)

        # ---- self-attn (this is what we cache) ----
        cond_normed = block.layer_norm_self_attn(cond_x) * (1 + sc_sa) + s_sa
        cq, ck, cv = _cond_qkv(attn, cond_normed, state.lora_qkv[idx], eff, cond_rope)
        cache.append((ck.detach(), cv.detach()))

        # ---- evolve cond_x to feed the next block ----
        attn_out = _self_attn_bshd(cq, ck, cv)
        attn_out = rearrange(attn_out, "b s h d -> b s (h d)")
        proj = attn.output_proj(attn_out)
        if state.lora_o[idx] is not None:
            proj = proj + eff * state.lora_o[idx](attn_out)
        cond_x = cond_x + g_sa * proj

        # ---- MLP ----
        mlp_normed = block.layer_norm_mlp(cond_x) * (1 + sc_mlp) + s_mlp
        hmid = block.mlp.layer1(mlp_normed)
        if state.lora_ffn1[idx] is not None:
            hmid = hmid + eff * state.lora_ffn1[idx](mlp_normed)
        hmid = block.mlp.activation(hmid)
        mlp_out = block.mlp.layer2(hmid)
        if state.lora_ffn2[idx] is not None:
            mlp_out = mlp_out + eff * state.lora_ffn2[idx](hmid)
        cond_x = cond_x + g_mlp * mlp_out

    state.cond_kv_cache = cache
    state._cache_device = device
    nbytes = sum(k.numel() + v.numel() for k, v in cache) * cache[0][0].element_size()
    logger.info(
        f"EasyControl: built cond KV cache ({len(cache)} blocks, "
        f"S_c={cache[0][0].shape[1]}, {nbytes / 1e6:.0f} MB)"
    )


# ---------------------------------------------------------------------------
# Extended self-attention (target keys + cached cond keys, b_cond gate)
# ---------------------------------------------------------------------------


def _extended_self_attn(tq, tk, tv, ck, cv, b_cond):
    """Attention of target queries over [target_k; cond_k] with an additive
    ``b_cond`` logit bias on the cond columns. All inputs [B, S, H, D] (cond may
    be batch-1 and is broadcast). Returns [B, S, H, D].

    Dispatches to the flash-attn LSE-decomposed path when enabled (see
    ``_extended_self_attn_flash``); otherwise uses the portable float-mask SDPA
    path. The float ``attn_mask`` forces SDPA onto its mem-efficient/math
    backend (flash SDPA can't take arbitrary float masks), so the flash path is
    a genuine speedup when available."""
    if (
        _flash_ext_enabled()
        and tq.is_cuda
        and tq.dtype in (torch.float16, torch.bfloat16)
    ):
        return _extended_self_attn_flash(tq, tk, tv, ck, cv, b_cond)

    B, S = tq.shape[0], tq.shape[1]
    if ck.shape[0] != B:
        ck = ck.expand(B, -1, -1, -1)
        cv = cv.expand(B, -1, -1, -1)
    Sc = ck.shape[1]
    k = torch.cat([tk, ck], dim=1).transpose(1, 2)  # [B,H,S+Sc,D]
    v = torch.cat([tv, cv], dim=1).transpose(1, 2)
    q = tq.transpose(1, 2)
    mask = torch.zeros(S + Sc, device=q.device, dtype=q.dtype)
    mask[S:] = b_cond.to(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.view(1, 1, 1, S + Sc))
    return out.transpose(1, 2)  # [B,S,H,D]


def _extended_self_attn_flash(tq, tk, tv, ck, cv, b_cond):
    """Flash-attn equivalent of ``_extended_self_attn`` via LSE decomposition.

    The joint softmax over ``[target_k ; cond_k + b_cond]`` is split into two
    independent flash attentions — one over target keys, one over cond keys —
    then recombined by log-sum-exp. The uniform scalar ``b_cond`` bias on the
    cond group is exact under this split: it cancels inside the cond group's own
    normalization (so ``out_c`` is computed bias-free) and re-enters only as a
    constant shift ``+b_cond`` on the cond group's LSE during the combine. This
    is the forward half of anima_lora ``_ExtendedSelfAttnLSEFunc``.

    flash_attn_func consumes ``[B, S, H, D]`` directly (no transpose) and
    returns ``softmax_lse`` as ``[B, H, S]`` in the natural-log domain with the
    softmax_scale already applied — matching SDPA's default ``1/sqrt(D)`` scale,
    so the two paths are numerically equivalent."""
    B = tq.shape[0]
    if ck.shape[0] != B:
        # flash requires matching batch dims; expand+contiguous (B is tiny — the
        # CFG fan, typically 1–2 — so the cond-KV duplication is cheap).
        ck = ck.expand(B, -1, -1, -1).contiguous()
        cv = cv.expand(B, -1, -1, -1).contiguous()

    out_t, lse_t, _ = _flash_attn_func(tq, tk, tv, return_attn_probs=True)
    out_c, lse_c, _ = _flash_attn_func(tq, ck, cv, return_attn_probs=True)

    # lse_*: [B, H, S], fp32. Shift the cond group by the trained scalar gate.
    lse_c_b = lse_c + b_cond.to(lse_c.dtype)
    log_z = torch.logaddexp(lse_t, lse_c_b)
    w_t = torch.exp(lse_t - log_z)  # [B, H, S]
    w_c = torch.exp(lse_c_b - log_z)
    # weights → [B, S, H, 1] to scale the [B, S, H, D] flash outputs.
    w_t = w_t.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
    w_c = w_c.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
    return out_t * w_t + out_c * w_c  # [B, S, H, D]


def _make_self_attn_forward(state: EasyControlState, idx: int):
    """Replacement for one block's ``self_attn.forward`` (an ``Attention``).

    Overriding only the self-attention — not the whole Block — keeps every other
    part of ComfyUI's block native (AdaLN, cross-attn, MLP, residuals, dtype
    handling), so there is nothing to drift and no block-level weights to strand
    (``Attention`` holds no direct parameters; its q/k/v/o projections and norms
    are submodules that keep their own cast-weights forward).

    Resolves the live ``Attention`` from ``state.live_blocks[idx].self_attn``
    each call, so a downstream DiT rebuild can't strand the patch on a dead
    instance.

    ``torch._dynamo.disable`` keeps this an opaque eager region so the closure
    composes with per-block ``torch.compile`` (e.g. the Anima Block Compile
    node). Without it, dynamo tries to trace the ``state.live_blocks[idx]``
    deref while compiling that very block and re-enters the module tree from a
    non-local source, tripping ``"UnspecializedNNModuleVariable(Block) is
    already tracked for mutation"``. The same live-block indirection that makes
    this patch survive a DiT rebuild is what dynamo can't reconcile, so we just
    graph-break around the extended attention (the rest of the block still
    compiles).
    """

    @torch._dynamo.disable
    def forward(x, context=None, rope_emb=None, transformer_options={}):
        attn = state.live_blocks[idx].self_attn
        cache = state.cond_kv_cache
        # Cond not ready, or this somehow ran as cross-attn → native attention.
        if cache is None or context is not None:
            return type(attn).forward(attn, x, context, rope_emb, transformer_options)

        tq, tk, tv = attn.compute_qkv(x, None, rope_emb=rope_emb)  # [B,S,H,D]
        ck, cv = cache[idx]
        out = _extended_self_attn(
            tq, tk, tv, ck.to(tk.dtype), cv.to(tv.dtype), state.b_cond[idx]
        )
        out = rearrange(out, "b s h d -> b s (h d)")
        return attn.output_dropout(attn.output_proj(out))

    return forward


# ---------------------------------------------------------------------------
# Pre-hook: lazy cond-KV prefill + live-block refresh
# ---------------------------------------------------------------------------


def _make_prefill_pre_hook(state: EasyControlState):
    def pre_hook(module, args):
        # module = live diffusion_model; args[0] = x (model-space latent).
        state.live_blocks = module.blocks
        x = args[0]
        device, dtype = x.device, x.dtype
        if state.cond_kv_cache is None or state._cache_device != device:
            _build_cond_kv(module, state, device, dtype)
        return None

    return pre_hook


# ---------------------------------------------------------------------------
# Public entry point — called by the node
# ---------------------------------------------------------------------------


def install_easycontrol(model, weight_path, cond_latent, strength, cond_scale_override=None):
    """Install EasyControl onto a cloned ModelPatcher.

    ``cond_latent`` is the VAE-encoded reference in **model space** (the node
    applies ``process_latent_in``), 4D ``[1, C, H, W]``.
    """
    sd, meta = _load_weights_and_meta(weight_path)
    state = EasyControlState(sd, meta, strength, cond_scale_override)
    state.cond_latent = cond_latent

    diffusion_model = model.get_model_object("diffusion_model")
    nblocks = len(diffusion_model.blocks)
    if nblocks != state.num_blocks:
        raise ValueError(
            f"EasyControl checkpoint is for {state.num_blocks} blocks but the "
            f"loaded DiT has {nblocks}. Wrong base model?"
        )

    # Lazy-prefill pre-hook on diffusion_model (object-patched OrderedDict so it
    # reverts on unpatch and composes with other pre-hooks).
    pre_hook = _make_prefill_pre_hook(state)
    new_pre = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre[id(pre_hook)] = pre_hook
    model.add_object_patch("diffusion_model._forward_pre_hooks", new_pre)

    # Per-block self-attention replacement (object-patched → path-resolved to
    # the live blocks; reverted on unpatch). Only self_attn is swapped; the rest
    # of each block stays native.
    for idx in range(state.num_blocks):
        model.add_object_patch(
            f"diffusion_model.blocks.{idx}.self_attn.forward",
            _make_self_attn_forward(state, idx),
        )

    # Keep the state alive for the patcher's lifetime.
    if not hasattr(model, "_easycontrol_states"):
        model._easycontrol_states = []
    model._easycontrol_states.append(state)
    logger.info(
        f"EasyControl: installed ({state.num_blocks} blocks, r={state.r}, "
        f"ffn_lora={state.apply_ffn}, eff_scale={state.eff_scale:.3f})"
    )
    return state
