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

import types

import torch

import comfy.ops as ops
from comfy.ldm.cosmos.predict2 import MiniTrainDIT

import easycontrol_patch as ec

# nodes.py imports `folder_paths` (a ComfyUI runtime module). Stub it so the
# mask-helper unit test can import the node module without a full ComfyUI runtime.
# Load by explicit path: a bare `import nodes` would resolve to ComfyUI's own
# root-level nodes.py (comfy root is on sys.path for the predict2 import above).
sys.modules.setdefault("folder_paths", types.SimpleNamespace(
    get_filename_list=lambda *a, **k: [],
    get_full_path=lambda *a, **k: None,
))
import importlib.util  # noqa: E402

_checkout = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# nodes.py does `from .easycontrol_patch import ...`, so it must load inside a
# package. Build a synthetic one rooted at the checkout dir.
_pkg = types.ModuleType("ec_pkg")
_pkg.__path__ = [_checkout]
sys.modules["ec_pkg"] = _pkg
for _sub in ("easycontrol_patch", "nodes"):
    _spec = importlib.util.spec_from_file_location(
        f"ec_pkg.{_sub}", os.path.join(_checkout, f"{_sub}.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"ec_pkg.{_sub}"] = _mod
    _spec.loader.exec_module(_mod)
ec_nodes = sys.modules["ec_pkg.nodes"]


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


def _stable_dit(use_adaln_lora=False):
    torch.manual_seed(0)
    dit = MiniTrainDIT(
        max_img_h=16, max_img_w=16, max_frames=1,
        in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=32, num_blocks=2, num_heads=2, mlp_ratio=4.0,
        crossattn_emb_channels=16,
        pos_emb_cls="rope3d", use_adaln_lora=use_adaln_lora,
        adaln_lora_dim=8,
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


def _build_sd_meta(num_blocks, D, ffn, r=4, adaln_dim=0, adaln_r=2, adaln_alpha=4.0):
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
        if adaln_dim:
            for branch in ("self_attn", "cross_attn", "mlp"):
                sd[f"adaln_lora_{branch}.{i}.lora_down.weight"] = (
                    torch.randn(adaln_r, adaln_dim) * 0.05
                )
                sd[f"adaln_lora_{branch}.{i}.lora_up.weight"] = (
                    torch.randn(3 * D, adaln_r) * 0.05
                )
    meta = {
        "ss_num_blocks": str(num_blocks), "ss_cond_lora_dim": str(r),
        "ss_cond_lora_alpha": str(r), "ss_apply_ffn_lora": "1", "ss_cond_scale": "1.0",
    }
    if adaln_dim:
        meta["ss_train_adaln"] = "1"
        meta["ss_adaln_alpha"] = str(adaln_alpha)
    return sd, meta


def _build_state(num_blocks, D, ffn, r=4, strength=1.0, **kw):
    sd, meta = _build_sd_meta(num_blocks, D, ffn, r=r, **kw)
    return ec.EasyControlState(sd, meta, strength=strength)


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


def test_inv_scale():
    """A channel-scaled checkpoint's ``inv_scale`` must multiply the input
    before the down GEMM (the trained function is ``up(down(x * inv))``)."""
    torch.manual_seed(3)
    D, r, out = 16, 4, 8
    down = torch.randn(r, D)
    up = torch.randn(out, r)
    inv = torch.rand(D) * 4 + 0.1
    x = torch.randn(2, D)
    lora = ec._LoRA(down.clone(), up.clone(), alpha_over_r=0.5, inv_scale=inv.clone())
    lora.to(torch.device("cpu"), torch.float32)
    ref = 0.5 * (x * inv) @ down.t() @ up.t()
    assert torch.allclose(lora(x), ref, atol=1e-5), "inv_scale must pre-scale the input"
    print("test_inv_scale PASSED")


def test_key_guard():
    """State construction must consume every checkpoint tensor: unknown keys
    raise (a silently dropped trained feature = garbage output), and adaln
    checkpoints must be complete."""
    D, ffn = 32, 128

    sd, meta = _build_sd_meta(2, D, ffn)
    sd["some_future_feature.0.weight"] = torch.zeros(3)
    try:
        ec.EasyControlState(sd, meta, strength=1.0)
        raise AssertionError("unknown key must raise")
    except ValueError as e:
        assert "some_future_feature" in str(e)

    # inv_scale keys are known (consumed), not "unknown".
    sd, meta = _build_sd_meta(2, D, ffn)
    sd["cond_lora_qkv.0.inv_scale"] = torch.ones(D)
    state = ec.EasyControlState(sd, meta, strength=1.0)
    assert state.lora_qkv[0].inv_scale is not None

    # Metadata says train_adaln but tensors are missing → truncated checkpoint.
    sd, meta = _build_sd_meta(2, D, ffn)
    meta["ss_train_adaln"] = "1"
    try:
        ec.EasyControlState(sd, meta, strength=1.0)
        raise AssertionError("ss_train_adaln without tensors must raise")
    except ValueError as e:
        assert "truncated" in str(e)

    # Partial adaln coverage → truncated checkpoint.
    sd, meta = _build_sd_meta(2, D, ffn, adaln_dim=8)
    del sd["adaln_lora_mlp.1.lora_down.weight"]
    del sd["adaln_lora_mlp.1.lora_up.weight"]
    try:
        ec.EasyControlState(sd, meta, strength=1.0)
        raise AssertionError("partial adaln must raise")
    except ValueError as e:
        assert "adaln_lora_mlp" in str(e)
    print("test_key_guard PASSED")


def test_adaln_merge_and_cond_subtract():
    """The two halves of adaln handling must cancel exactly:

    - target stream: merging ``strength * (alpha/r) * up @ down`` onto
      ``adaln_modulation_*.2.weight`` (what the ModelPatcher patch computes)
      equals the training-side ``adaln_up(h) + strength * delta(h)``;
    - cond stream: ``_cond_branch_modulation`` on the *merged* weights
      reproduces the un-merged (frozen) modulation — training never applies the
      adaln delta on the cond side.
    """
    torch.manual_seed(4)
    dit = _stable_dit(use_adaln_lora=True)
    D = dit.model_channels
    adaln_dim = dit.blocks[0].adaln_modulation_self_attn[1].weight.shape[0]
    strength = 0.7
    state = _build_state(len(dit.blocks), D, int(D * 4.0),
                         adaln_dim=adaln_dim, strength=strength)
    assert state.train_adaln and state.adaln_rank == 2

    emb = torch.randn(1, 1, D)
    for idx, block in enumerate(dit.blocks):
        for branch in ("self_attn", "cross_attn", "mlp"):
            seq = getattr(block, f"adaln_modulation_{branch}")
            delta = state.adaln[branch][idx]
            base_out = seq(emb)

            # Merge the patch the way comfy's LoRAAdapter does (strength * α/r * up@down).
            merged_w = (
                seq[2].weight
                + strength * delta.scale * (delta.up.float() @ delta.down.float())
            )
            h = seq[1](seq[0](emb))
            target_merged = h @ merged_w.t()
            target_train = seq[2](h) + strength * delta(h)  # training-side form
            assert torch.allclose(target_merged, target_train, atol=1e-5), (
                f"merged patch != training delta on {branch}.{idx}"
            )

            # Cond stream: subtract on merged weights recovers the frozen base.
            orig = seq[2].weight.data.clone()
            seq[2].weight.data.copy_(merged_w)
            try:
                cond_out = ec._cond_branch_modulation(seq, emb, delta, strength)
            finally:
                seq[2].weight.data.copy_(orig)
            assert torch.allclose(cond_out, base_out, atol=1e-5), (
                f"cond subtract must recover frozen modulation on {branch}.{idx}"
            )
    print("test_adaln_merge_and_cond_subtract PASSED")


def test_adaln_plumbing():
    """End-to-end wiring on an adaln-lora DiT: cond-KV prefill (which exercises
    the subtract path) + full patched forward stay finite."""
    dit = _stable_dit(use_adaln_lora=True)
    D = dit.model_channels
    adaln_dim = dit.blocks[0].adaln_modulation_self_attn[1].weight.shape[0]
    state = _build_state(len(dit.blocks), D, int(D * 4.0), adaln_dim=adaln_dim)

    state.cond_latent = torch.randn(1, 4, 8, 8) * 0.5
    state.live_blocks = dit.blocks
    ec._build_cond_kv(dit, state, torch.device("cpu"), torch.float32)
    assert state.cond_kv_cache is not None
    for k, v in state.cond_kv_cache:
        assert torch.isfinite(k).all() and torch.isfinite(v).all()

    for idx in range(len(dit.blocks)):
        dit.blocks[idx].self_attn.forward = ec._make_self_attn_forward(state, idx)
    x = torch.randn(1, 4, 1, 8, 8) * 0.5
    with torch.no_grad():
        out = dit(x, torch.tensor([0.5]), torch.randn(1, 7, 16) * 0.5)
    assert out.shape == (1, 4, 1, 8, 8) and torch.isfinite(out).all()
    print("test_adaln_plumbing PASSED")


def test_inpaint_mask():
    """`_apply_inpaint_mask` gray-fills the selected region and leaves the rest
    pixel-identical, resizing a mismatched mask to the image grid."""
    gray = ec_nodes._INPAINT_GRAY
    pixels = torch.rand(1, 8, 8, 3)

    # Same-size mask: left half selected (hole), right half kept.
    mask = torch.zeros(1, 8, 8)
    mask[:, :, :4] = 1.0
    out = ec_nodes._apply_inpaint_mask(pixels, mask)
    assert torch.allclose(out[:, :, :4, :], torch.full_like(out[:, :, :4, :], gray))
    assert torch.allclose(out[:, :, 4:, :], pixels[:, :, 4:, :]), "kept region must be untouched"

    # Mismatched mask resolution must be resized to the image grid (no crash, hole present).
    coarse = torch.zeros(1, 2, 2)
    coarse[:, 0, 0] = 1.0
    out2 = ec_nodes._apply_inpaint_mask(pixels, coarse)
    assert torch.allclose(out2[:, :4, :4, :], torch.full_like(out2[:, :4, :4, :], gray))
    assert torch.allclose(out2[:, 4:, 4:, :], pixels[:, 4:, 4:, :])
    print("test_inpaint_mask PASSED")


if __name__ == "__main__":
    test_extended_attn_math()
    test_plumbing()
    test_inv_scale()
    test_key_guard()
    test_adaln_merge_and_cond_subtract()
    test_adaln_plumbing()
    test_inpaint_mask()
    print("\nALL CHECKS PASSED")
