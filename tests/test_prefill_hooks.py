"""Regression checks for EasyControl pre-hook replacement.

These tests intentionally cover only the hook-registration contract.  They do
not require a checkpoint, a GPU, or a real ComfyUI ModelPatcher.
"""

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CHECKOUT = Path(__file__).resolve().parents[1]
COMFY_ROOT = CHECKOUT.parents[1]
sys.path.insert(0, str(CHECKOUT))
sys.path.insert(0, str(COMFY_ROOT))

import easycontrol_patch as ec  # noqa: E402


class _FakeDiffusionModel:
    def __init__(self, hooks):
        self._forward_pre_hooks = OrderedDict(hooks)
        self.blocks = [SimpleNamespace(x_dim=1)]


class _FakeModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.object_patches = []

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion_model

    def add_object_patch(self, path, value):
        self.object_patches.append((path, value))

    def add_wrapper_with_key(self, key, name, wrapper):
        pass


def _install_with_hooks(existing_hooks):
    diffusion_model = _FakeDiffusionModel(existing_hooks)
    model = _FakeModel(diffusion_model)
    state = SimpleNamespace(
        num_blocks=1,
        hidden_dim=0,
        train_adaln=False,
        cond_latent=None,
        r=0,
        apply_ffn=False,
        eff_scale=1.0,
    )

    with patch.object(ec, "_load_weights_and_meta", return_value=({}, {})), \
            patch.object(ec, "EasyControlState", return_value=state):
        installed = ec.install_easycontrol(model, "unused", None, 1.0)

    assert installed is state
    hook_patches = [
        value for path, value in model.object_patches
        if path == "diffusion_model._forward_pre_hooks"
    ]
    assert len(hook_patches) == 1
    return hook_patches[0]


def test_prefill_hook_is_marked_as_easycontrol():
    hook = ec._make_prefill_pre_hook(SimpleNamespace())

    assert getattr(hook, "_easycontrol_prefill_hook", False) is True


def test_reinstall_replaces_easycontrol_hooks_and_preserves_other_hooks():
    old_state = SimpleNamespace()
    old_hook = ec._make_prefill_pre_hook(old_state)

    def other_hook(module, args):
        return None

    new_hooks = _install_with_hooks([
        ("other", other_hook),
        ("old-easycontrol", old_hook),
    ])

    assert list(new_hooks) == ["other", next(iter(set(new_hooks) - {"other"}))]
    assert new_hooks["other"] is other_hook
    assert old_hook not in new_hooks.values()

    easycontrol_hooks = [
        hook for hook in new_hooks.values()
        if getattr(hook, "_easycontrol_prefill_hook", False)
    ]
    assert len(easycontrol_hooks) == 1


if __name__ == "__main__":
    test_prefill_hook_is_marked_as_easycontrol()
    test_reinstall_replaces_easycontrol_hooks_and_preserves_other_hooks()
    print("ALL PREFILL HOOK CHECKS PASSED")
