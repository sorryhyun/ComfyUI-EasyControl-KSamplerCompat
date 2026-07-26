"""EasyEdit delta-caption builders — CPU-only prompt helpers, no dependencies.

The ``anima_subject_edit`` EasyControl adapter is trained on *instructions*, not
descriptions: the prompt is the tag delta between the conditioning image's
caption and the desired result's caption —

    additions (bare tags, in the target caption's order)
    then removals (the source's tags that should go, each prefixed with ``-``)

Shared tags cancel out, which is the whole point: the character name drops out
of the prompt, so identity can only come from the reference image. Hand-writing
that string is easy to get subtly wrong (removal prefix, ordering, a tag left in
both halves), so these two nodes build it:

``AnimaEasyEditInstruction``
    Two text fields — what to add, what to remove — into one instruction.

``AnimaEasyEditDelta``
    Two full captions — the reference's and your edited copy of it — into the
    delta between them. Mirrors ``easycontrol_adapters/tools/
    subject_edit_pairs.py::delta_caption`` (the miner that produced the training
    prompts) exactly, so what the node emits is shaped like what the adapter
    saw. Wire the Anima Tagger's caption output into ``source_caption``, edit a
    copy into ``target_caption``, and you get "edit by editing the caption"
    without an inversion pass.
"""

# Marks a tag as "remove this" in a delta caption. Matches the miner's
# --removal_prefix default; the adapter learned negation against this exact
# token, so it is a constant here rather than a user knob.
REMOVAL_PREFIX = "-"

# Underscore→space normalization is skipped for short tags so Danbooru
# emoticons (``^_^``, ``>_<``, ``x_x``, ``._.`` — all 3 chars) survive intact,
# while ordinary underscored tags (``blue_eyes``) are converted to the
# space-separated form Anima's vocabulary actually uses.
_EMOTICON_MAX_LEN = 3


def _normalize(tag: str, underscores_to_spaces: bool) -> str:
    tag = tag.strip()
    if (
        underscores_to_spaces
        and "_" in tag
        and len(tag) > _EMOTICON_MAX_LEN
        and any(c.isalpha() for c in tag)
    ):
        tag = tag.replace("_", " ").strip()
    return tag


def split_tags(text: str, underscores_to_spaces: bool = True) -> list[str]:
    """A caption / tag-list string → an ordered, de-duplicated tag list.

    Commas and newlines both separate (users paste one-per-line as often as
    comma-joined). Empty fragments are dropped; the first occurrence of a
    repeated tag wins, so order is stable.
    """
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for chunk in text.replace("\n", ",").split(","):
        tag = _normalize(chunk, underscores_to_spaces)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def build_instruction(additions: list[str], removals: list[str]) -> str:
    """(additions, removals) → the delta-caption instruction string.

    Additions come first as bare tags, then removals prefixed with
    ``REMOVAL_PREFIX``. A tag already carrying the prefix is not double-
    prefixed, and a tag present in *both* lists is treated as an addition only
    (a "add X and remove X" instruction is contradictory; the adapter follows
    additions far more reliably, so that is the safe reading).
    """
    add = [t.lstrip(REMOVAL_PREFIX).strip() for t in additions]
    add = [t for t in add if t]
    add_set = set(add)
    rem = []
    for t in removals:
        bare = t.lstrip(REMOVAL_PREFIX).strip()
        if bare and bare not in add_set:
            rem.append(f"{REMOVAL_PREFIX}{bare}")
    return ", ".join(add + rem)


def caption_delta(
    src_tags: list[str], dst_tags: list[str], include_removals: bool = True
) -> tuple[list[str], list[str]]:
    """(additions, removals) between two tag lists.

    Additions in the target caption's order, removals in the source's — the
    same convention the pair miner used to write the training prompts.
    """
    src, dst = set(src_tags), set(dst_tags)
    additions = [t for t in dst_tags if t not in src]
    removals = [t for t in src_tags if t not in dst] if include_removals else []
    return additions, removals


_UNDERSCORE_TOOLTIP = (
    "Convert underscored tags to the space-separated form Anima's vocabulary "
    "uses (blue_eyes -> blue eyes). Short emoticon tags (^_^, x_x) are left "
    "alone. Turn off if you are pasting tags that are already normalized."
)


class AnimaEasyEditInstruction:
    """Build an EasyEdit instruction from separate add / remove tag lists."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "additions": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Tags to ADD, comma- or newline-separated. What "
                            "should be true of the result but is not true of "
                            "the reference image: 'wading in water, twintails, "
                            "night'. Anima tag vocabulary."
                        ),
                    },
                ),
                "removals": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Tags to REMOVE, comma- or newline-separated, "
                            "written WITHOUT the '-' prefix (the node adds it). "
                            "These must be tags that describe the REFERENCE "
                            "image. Note: removing an object that is physically "
                            "present in the reference is the adapter's weakest "
                            "case — see the README."
                        ),
                    },
                ),
            },
            "optional": {
                "underscores_to_spaces": (
                    "BOOLEAN",
                    {"default": True, "tooltip": _UNDERSCORE_TOOLTIP},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instruction",)
    FUNCTION = "build"
    CATEGORY = "utils"
    DESCRIPTION = (
        "Build an Anima EasyEdit delta-caption instruction from two tag lists "
        "(add / remove). Emits 'additions..., -removals...' — the prompt "
        "convention the subject_edit EasyControl adapter was trained on. Feed "
        "the result into your positive text encode; feed the reference image "
        "into Anima EasyControl (KSampler)."
    )

    def build(self, additions, removals, underscores_to_spaces=True):
        add = split_tags(additions, underscores_to_spaces)
        rem = split_tags(removals, underscores_to_spaces)
        return (build_instruction(add, rem),)


class AnimaEasyEditDelta:
    """Compute the EasyEdit instruction as the delta between two captions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "The REFERENCE image's caption. Wire the Anima "
                            "Tagger's STRING output here, or paste the tags "
                            "yourself."
                        ),
                    },
                ),
                "target_caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "The caption you WANT — normally a copy of "
                            "source_caption with tags edited, added, or "
                            "deleted. The node emits only the difference, so "
                            "everything you left alone (including the "
                            "character name) drops out of the prompt."
                        ),
                    },
                ),
            },
            "optional": {
                "include_removals": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Emit source-only tags as '-tag' removals. Off = "
                            "additions only, which is a weaker but safer "
                            "instruction when the two captions differ in ways "
                            "you did not intend to change."
                        ),
                    },
                ),
                "underscores_to_spaces": (
                    "BOOLEAN",
                    {"default": True, "tooltip": _UNDERSCORE_TOOLTIP},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("instruction", "additions", "removals")
    FUNCTION = "diff"
    CATEGORY = "utils"
    DESCRIPTION = (
        "Diff two Anima captions into an EasyEdit delta-caption instruction: "
        "tags only in the target become additions, tags only in the source "
        "become '-' removals, shared tags cancel. Mirrors the pair miner that "
        "generated the subject_edit adapter's training prompts. Pair with the "
        "Anima Tagger to edit an image by editing its caption."
    )

    def diff(
        self,
        source_caption,
        target_caption,
        include_removals=True,
        underscores_to_spaces=True,
    ):
        src = split_tags(source_caption, underscores_to_spaces)
        dst = split_tags(target_caption, underscores_to_spaces)
        additions, removals = caption_delta(src, dst, include_removals)
        return (
            build_instruction(additions, removals),
            ", ".join(additions),
            ", ".join(removals),
        )


NODE_CLASS_MAPPINGS = {
    "AnimaEasyEditInstruction": AnimaEasyEditInstruction,
    "AnimaEasyEditDelta": AnimaEasyEditDelta,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaEasyEditInstruction": "Anima EasyEdit Instruction (add / remove)",
    "AnimaEasyEditDelta": "Anima EasyEdit Delta (caption diff)",
}
