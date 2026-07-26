"""Checks for the EasyEdit delta-caption builders (pure Python, no ComfyUI).

The load-bearing claim is *equivalence with the training-time miner*: the
prompts the adapter learned from were produced by
``easycontrol_adapters/tools/subject_edit_pairs.py::delta_caption``, so a node
that emits a differently-ordered or differently-prefixed string is feeding the
adapter something it never saw. ``test_matches_miner`` reimplements the miner's
function inline (it lives in the anima_lora repo, not importable here) and
asserts byte-equality on the delta.

Run:  python tests/test_delta_caption.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta_caption import (  # noqa: E402
    REMOVAL_PREFIX,
    AnimaEasyEditDelta,
    AnimaEasyEditInstruction,
    build_instruction,
    caption_delta,
    split_tags,
)


def _miner_delta_caption(src_tags, dst_tags, removal_prefix, include_removals):
    """Verbatim copy of the training-time miner's delta_caption()."""
    src, dst = set(src_tags), set(dst_tags)
    additions = [t for t in dst_tags if t not in src]
    removals = [t for t in src_tags if t not in dst]
    tags = list(additions)
    if include_removals:
        tags += [f"{removal_prefix}{t}" for t in removals]
    return tags, len(additions), len(removals)


def test_split_tags():
    assert split_tags("1girl, halo, blue eyes") == ["1girl", "halo", "blue eyes"]
    # Newlines separate too, whitespace is stripped, empties dropped.
    assert split_tags("1girl,\n  halo ,,\nsolo\n") == ["1girl", "halo", "solo"]
    # De-duplicated, first occurrence wins (order stable).
    assert split_tags("halo, solo, halo") == ["halo", "solo"]
    assert split_tags("") == []
    # Underscore normalization: real tags converted, emoticons preserved.
    assert split_tags("blue_eyes, ^_^, x_x, ._.") == ["blue eyes", "^_^", "x_x", "._."]
    assert split_tags("blue_eyes", underscores_to_spaces=False) == ["blue_eyes"]
    print("test_split_tags PASSED")


def test_build_instruction():
    got = build_instruction(["wading in water", "night"], ["chair", "ramune"])
    assert got == "wading in water, night, -chair, -ramune", got

    # A '-' the user typed themselves must not be doubled.
    assert build_instruction([], ["-hat"]) == "-hat"
    # ...and a '-' typed in the additions field is stripped, not emitted.
    assert build_instruction(["-hat"], []) == "hat"

    # Contradictory tag (in both lists) resolves to the addition only.
    assert build_instruction(["halo"], ["halo", "hat"]) == "halo, -hat"

    # Empty halves are legal: addition-only and removal-only instructions.
    assert build_instruction(["smile"], []) == "smile"
    assert build_instruction([], ["smile"]) == "-smile"
    assert build_instruction([], []) == ""
    print("test_build_instruction PASSED")


def test_caption_delta_ordering():
    src = ["safe", "1girl", "izuna", "chair", "ramune", "halo"]
    dst = ["safe", "1girl", "izuna", "wading in water", "halo", "night"]
    additions, removals = caption_delta(src, dst)
    # Additions in the TARGET's order; removals in the SOURCE's.
    assert additions == ["wading in water", "night"], additions
    assert removals == ["chair", "ramune"], removals
    # Shared tags — including the character name — cancel out entirely.
    assert "izuna" not in additions + removals

    _, no_removals = caption_delta(src, dst, include_removals=False)
    assert no_removals == []
    print("test_caption_delta_ordering PASSED")


def test_matches_miner():
    """The node's output must equal the training-time miner's, byte for byte."""
    cases = [
        (
            ["safe", "1girl", "hoshino", "jacket", "beach"],
            ["safe", "1girl", "hoshino", "bikini", "sunglasses", "beach"],
        ),
        (["a", "b", "c"], ["c", "b", "a"]),  # pure reorder → empty delta
        (["a"], ["b"]),
        ([], ["x", "y"]),
        (["x", "y"], []),
    ]
    for src, dst in cases:
        for include_removals in (True, False):
            expect, n_add, n_rem = _miner_delta_caption(
                src, dst, REMOVAL_PREFIX, include_removals
            )
            additions, removals = caption_delta(src, dst, include_removals)
            got = build_instruction(additions, removals)
            assert got == ", ".join(expect), f"{src}->{dst}: {got!r} != {expect!r}"
            assert len(additions) == n_add
            # The miner reports n_rem as a corpus stat even when it suppresses
            # the removal tags, so only the emitting case is comparable.
            if include_removals:
                assert len(removals) == n_rem
    print("test_matches_miner PASSED")


def test_node_surfaces():
    """The ComfyUI-facing methods return the tuples their RETURN_TYPES declare."""
    instr = AnimaEasyEditInstruction()
    out = instr.build("wading in water", "chair")
    assert out == ("wading in water, -chair",), out
    assert len(out) == len(AnimaEasyEditInstruction.RETURN_TYPES)

    delta = AnimaEasyEditDelta()
    out = delta.diff("safe, 1girl, izuna, chair", "safe, 1girl, izuna, night")
    assert out == ("night, -chair", "night", "chair"), out
    assert len(out) == len(AnimaEasyEditDelta.RETURN_TYPES)

    # A tagger caption pasted unchanged into both fields = a no-op instruction,
    # not a crash — the user has not asked for anything yet.
    caption = "safe, 1girl, izuna (blue archive), halo, chair"
    assert delta.diff(caption, caption) == ("", "", "")

    # Every declared input name must be accepted by the FUNCTION.
    for cls in (AnimaEasyEditInstruction, AnimaEasyEditDelta):
        spec = cls.INPUT_TYPES()
        names = list(spec["required"]) + list(spec.get("optional", {}))
        fn = getattr(cls(), cls.FUNCTION)
        kwargs = {n: "" if n.endswith("caption") or n in ("additions", "removals") else True for n in names}
        fn(**kwargs)
    print("test_node_surfaces PASSED")


if __name__ == "__main__":
    test_split_tags()
    test_build_instruction()
    test_caption_delta_ordering()
    test_matches_miner()
    test_node_surfaces()
    print("\nALL CHECKS PASSED")
