"""Numeric checks for the EasyControl KSampler-compat reimplementation.

Two independent checks, no ComfyUI runtime / no Anima checkpoint needed:

  test_extended_attn_math
      The core correctness claim, tested directly on the attention math:
        - with a very negative b_cond the cond keys are gated out, so extended
          attention == plain target-only self-attention (the step-0 equivalence
          that makes the splice safe to drop on a trained model);
        - with b_cond = 0 the cond keys contribute, so the output differs;
        - the cond batch (1) broadcasts to the target batch.

  test_plumbing
      End-to-end wiring on a tiny real predict2 DiT (the backbone Anima
      inherits): build the cond-KV cache, override each block's self_attn, run a
      full DiT forward, and assert it's finite and shape-preserving. The DiT is
      stabilized (adaln gates near-zero) only so an *untrained* net doesn't blow
      up — this check is about wiring, not values.

Run:  python tests/test_smoke.py   (with the comfy repo importable)
"""

import os
import sys

# Make `comfy` importable: prefer an already-installed comfy, else COMFY_PATH,
# else a couple of common dev locations relative to this checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import comfy.ops  # noqa: F401
except ImportError:
    for cand in (
        os.environ.get("COMFY_PATH"),
        os.path.expanduser("~/anima/comfy"),
        os.path.expanduser("~/comfy"),
    ):
        if cand and os.path.isdir(os.path.join(cand, "comfy")):
            sys.path.insert(0, cand)
            break

import torch

import comfy.ops as ops
from comfy.ldm.cosmos.predict2 import MiniTrainDIT

import easycontrol_patch as ec


def test_extended_attn_math():
    torch.manual_seed(0)
    B, S, Sc, H, D = 2, 12, 5, 3, 16
    tq = torch.randn(B, S, H, D)
    tk = torch.randn(B, S, H, D)
    tv = torch.randn(B, S, H, D)
    ck = torch.randn(1, Sc, H, D)  # cond is batch-1 → must broadcast
    cv = torch.randn(1, Sc, H, D)

    # Reference: plain target-only self-attention.
    base = torch.nn.functional.scaled_dot_product_attention(
        tq.transpose(1, 2), tk.transpose(1, 2), tv.transpose(1, 2)
    ).transpose(1, 2)

    gated = ec._extended_self_attn(tq, tk, tv, ck, cv, torch.tensor(-1e4))
    rel = (gated - base).abs().max().item() / base.abs().max().item()
    print(f"  [math] gated rel-diff vs target-only = {rel:.2e}")
    assert gated.shape == base.shape
    assert rel < 1e-5, f"very-negative b_cond must reduce to target-only, rel={rel}"

    active = ec._extended_self_attn(tq, tk, tv, ck, cv, torch.tensor(0.0))
    delta = (active - base).abs().max().item()
    print(f"  [math] active abs-diff vs target-only = {delta:.2e}")
    assert delta > 1e-2, "b_cond=0 must let cond keys change the output"
    assert torch.isfinite(active).all()
    print("test_extended_attn_math PASSED")


def _stable_dit():
    torch.manual_seed(0)
    dit = MiniTrainDIT(
        max_img_h=16, max_img_w=16, max_frames=1,
        in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=32, num_blocks=2, num_heads=2, mlp_ratio=4.0,
        crossattn_emb_channels=16,
        pos_emb_cls="rope3d", use_adaln_lora=False,
        operations=ops.disable_weight_init,
    ).eval().float()
    # disable_weight_init leaves Linear weights UNINITIALIZED (ComfyUI expects a
    # checkpoint), so init everything deterministically: norm weights → 1, all
    # other weights → small normal. Then shrink the adaln_modulation output so an
    # untrained net's residual stream stays bounded.
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in dit.named_parameters():
            if p.ndim == 1:  # RMSNorm weights (and any bias)
                p.fill_(1.0 if "norm" in name else 0.0)
            else:
                p.normal_(0.0, 0.05)
        for blk in dit.blocks:
            for name in ("adaln_modulation_self_attn",
                         "adaln_modulation_cross_attn",
                         "adaln_modulation_mlp"):
                getattr(blk, name)[-1].weight.mul_(0.1)
    return dit


def _build_state(num_blocks, D, ffn, r=4):
    sd = {}
    for i in range(num_blocks):
        sd[f"cond_lora_qkv.{i}.lora_down.weight"] = torch.randn(r, D) * 0.05
        sd[f"cond_lora_qkv.{i}.lora_up.weight"] = torch.randn(3 * D, r) * 0.05
        sd[f"cond_lora_o.{i}.lora_down.weight"] = torch.randn(r, D) * 0.05
        sd[f"cond_lora_o.{i}.lora_up.weight"] = torch.randn(D, r) * 0.05
        sd[f"cond_lora_ffn1.{i}.lora_down.weight"] = torch.randn(r, D) * 0.05
        sd[f"cond_lora_ffn1.{i}.lora_up.weight"] = torch.randn(ffn, r) * 0.05
        sd[f"cond_lora_ffn2.{i}.lora_down.weight"] = torch.randn(r, ffn) * 0.05
        sd[f"cond_lora_ffn2.{i}.lora_up.weight"] = torch.randn(D, r) * 0.05
        sd[f"b_cond.{i}"] = torch.tensor(0.0)
    meta = {
        "ss_num_blocks": str(num_blocks), "ss_cond_lora_dim": str(r),
        "ss_cond_lora_alpha": str(r), "ss_apply_ffn_lora": "1", "ss_cond_scale": "1.0",
    }
    return ec.EasyControlState(sd, meta, strength=1.0)


def test_plumbing():
    dit = _stable_dit()
    D = dit.model_channels
    state = _build_state(len(dit.blocks), D, int(D * 4.0))

    # Build the cond-KV cache from a random reference latent.
    state.cond_latent = torch.randn(1, 4, 8, 8) * 0.5
    state.live_blocks = dit.blocks
    ec._build_cond_kv(dit, state, torch.device("cpu"), torch.float32)
    assert state.cond_kv_cache is not None and len(state.cond_kv_cache) == len(dit.blocks)
    for k, v in state.cond_kv_cache:
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
    print(f"  [plumb] cond KV cache: {len(state.cond_kv_cache)} blocks, "
          f"S_c={state.cond_kv_cache[0][0].shape[1]}")

    # Override each block's self_attn and run a full forward.
    for idx in range(len(dit.blocks)):
        dit.blocks[idx].self_attn.forward = ec._make_self_attn_forward(state, idx)

    x = torch.randn(1, 4, 1, 8, 8) * 0.5
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 7, 16) * 0.5
    with torch.no_grad():
        out = dit(x, t, ctx)
    assert out.shape == (1, 4, 1, 8, 8), out.shape
    assert torch.isfinite(out).all(), "patched DiT forward produced non-finite output"
    print(f"  [plumb] full patched forward: shape={tuple(out.shape)} finite=OK")
    print("test_plumbing PASSED")


if __name__ == "__main__":
    test_extended_attn_math()
    test_plumbing()
    print("\nALL CHECKS PASSED")
