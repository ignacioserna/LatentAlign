"""Expanded CLIP phrase bank for slice / group description.

`describe.build_phrase_bank()` ships ~55 phrases. That is enough to name a slice
but not enough to *discover* one: B2T ranks phrases by a CLIP-score gap, so its
resolution is bounded by how finely the bank tiles description space, and Domino's
MeanDescriber can only return a phrase that exists in the bank. With 55 phrases
the top-scoring phrase for two genuinely different groups is often the same one.

This module keeps `describe.build_phrase_bank()` as the head of the bank -- so the
descriptions that already appear in your report stay in, at the same indices --
and appends a much larger compositional expansion:

    head        the original ~55 phrases, order preserved
    axes        single-attribute phrases across 12 axes, each in 4 templates
    pairs       two-attribute compositions (quality x occlusion, pose x lighting,
                age x eyewear, ...), which is where the interesting slices live:
                "a blurry photo of an elderly person" is a failure mode,
                "a blurry photo" and "an elderly person" separately are not

Ordering is deterministic and the head is stable, so appending phrases never
invalidates a cached `text_embs.npy` prefix -- `expand_phrases.py` relies on this
to re-encode only the tail.

Race and ethnicity terms are deliberately excluded, following the reasoning in
`describe.py`: RFW gives exact race metadata, and CLIP race attribution is
unreliable enough that a bank containing it mostly produces confidently wrong
descriptions. Pass ``include_race=True`` if you want them for an ablation.
"""

from __future__ import annotations

import itertools

try:
    from .describe import build_phrase_bank as _original_bank
except ImportError:  # allow direct execution
    from describe import build_phrase_bank as _original_bank


# --------------------------------------------------------------------------
# single-attribute axes
# --------------------------------------------------------------------------

AXES = {
    "pose": [
        "a face looking straight at the camera",
        "a face turned slightly to the side",
        "a face at a three-quarter angle",
        "a full side profile of a face",
        "a face turned away from the camera",
        "a head tilted sharply to one side",
        "a face photographed from below",
        "a face photographed from above",
        "a person looking down",
        "a person looking away from the lens",
        "a face at an extreme yaw angle",
        "a perfectly frontal passport-style face",
    ],
    "lighting": [
        "a face in very low light",
        "a face lit from one side only",
        "a face with a hard shadow across it",
        "a face washed out by flash",
        "a strongly backlit face against a bright background",
        "a face under warm orange stage lighting",
        "a face under cold blue lighting",
        "a face under harsh overhead lighting",
        "a face in even diffuse daylight",
        "a face with specular highlights on the skin",
        "a face half in shadow and half in sunlight",
        "a face lit by a screen in a dark room",
    ],
    "quality": [
        "a sharp high-resolution photograph",
        "a heavily blurred photograph",
        "a motion-blurred photograph of a moving person",
        "an out-of-focus photograph",
        "a grainy high-ISO photograph",
        "a heavily compressed low-quality image",
        "a pixelated upscaled image",
        "a small low-resolution thumbnail",
        "a noisy photograph taken in the dark",
        "a photograph with visible JPEG artifacts",
        "a soft-focus retouched portrait",
        "a photograph with a colour cast",
        "a black and white photograph",
        "a faded old photograph",
    ],
    "eyewear": [
        "a person wearing no glasses",
        "a person wearing thin wire-framed glasses",
        "a person wearing thick black-framed glasses",
        "a person wearing rimless glasses",
        "a person wearing dark sunglasses",
        "a person wearing mirrored sunglasses",
        "a person wearing tinted glasses",
        "a person wearing reading glasses low on the nose",
        "a person wearing safety goggles",
        "a person whose glasses reflect the light",
    ],
    "facial_hair": [
        "a clean-shaven face",
        "a face with light stubble",
        "a face with heavy stubble",
        "a face with a short beard",
        "a face with a long full beard",
        "a face with a moustache only",
        "a face with a goatee",
        "a face with thick sideburns",
        "a face with a grey beard",
        "a face with a neatly trimmed beard",
    ],
    "age": [
        "a photograph of an infant",
        "a photograph of a young child",
        "a photograph of a teenager",
        "a photograph of a person in their twenties",
        "a photograph of a person in their thirties",
        "a photograph of a middle-aged person",
        "a photograph of an older person",
        "a photograph of an elderly person with deep wrinkles",
        "a photograph of a person with grey hair",
        "a youthful smooth-skinned face",
    ],
    "headwear": [
        "a person wearing a baseball cap",
        "a person wearing a wide-brimmed hat",
        "a person wearing a woollen beanie",
        "a person wearing a headscarf",
        "a person wearing a hijab",
        "a person wearing a turban",
        "a person wearing a helmet",
        "a person wearing a military cap",
        "a person wearing a hood over their head",
        "a person wearing a graduation cap",
        "a person with a cap brim shading the eyes",
        "a bare-headed person",
    ],
    "occlusion": [
        "a face partly covered by a hand",
        "a face behind a microphone",
        "a face partly hidden by hair",
        "a face wearing a surgical mask",
        "a face wrapped in a scarf",
        "a face behind a fence or railing",
        "a face partly out of frame",
        "a face behind glass with reflections",
        "a face obscured by an object in the foreground",
        "an unobstructed clearly visible face",
    ],
    "expression": [
        "a neutral unsmiling face",
        "a broad smile showing teeth",
        "a closed-lip smile",
        "a laughing face",
        "a face shouting or singing",
        "a frowning angry face",
        "a surprised face with raised eyebrows",
        "a face with the eyes closed",
        "a face mid-blink",
        "a crying or distressed face",
    ],
    "hair": [
        "a person with short hair",
        "a person with long hair",
        "a person with curly hair",
        "a person with straight hair",
        "a bald person",
        "a person with a receding hairline",
        "a person with hair tied back",
        "a person with a fringe covering the forehead",
        "a person with dyed bright hair",
        "a person with wet hair",
    ],
    "styling": [
        "a person wearing heavy makeup",
        "a person wearing no makeup",
        "a person wearing bright lipstick",
        "a person wearing large earrings",
        "a person wearing a suit and tie",
        "a person in a uniform",
        "a person wearing a high-collared garment",
        "a person with visible tattoos",
        "a person with facial piercings",
        "a person wearing a necklace",
    ],
    "context": [
        "a press photograph at an event",
        "a red carpet photograph",
        "a photograph taken on stage under spotlights",
        "a sports photograph on a field",
        "a studio portrait against a plain backdrop",
        "a candid outdoor snapshot",
        "a photograph with a crowded background",
        "a photograph taken at night",
        "a surveillance camera still",
        "a webcam screenshot",
        "a scanned printed photograph",
        "a group photo cropped to one person",
    ],
}

if False:  # kept visible, off by default -- see the module docstring
    AXES["race"] = []

RACE_TERMS = [
    "a photograph of a Black person",
    "a photograph of a White person",
    "a photograph of an East Asian person",
    "a photograph of a South Asian person",
]

TEMPLATES = [
    "{}",
    "a photo of {}",
    "a cropped photo of {}",
    "a close-up photograph showing {}",
]

# axis pairs whose *combination* is a plausible failure mode
COMPOSITIONS = [
    ("quality", "occlusion"),
    ("quality", "age"),
    ("pose", "lighting"),
    ("pose", "occlusion"),
    ("lighting", "facial_hair"),
    ("age", "eyewear"),
    ("headwear", "lighting"),
    ("quality", "pose"),
    ("expression", "occlusion"),
    ("age", "quality"),
]


def _fmt(template, term):
    """Apply a template without producing 'a photo of a photo of ...'."""
    if template == "{}":
        return term
    if term.startswith(("a photo", "a photograph", "a cropped", "a close-up",
                        "a press", "a red carpet", "a studio", "a candid",
                        "a scanned", "a group", "a sports", "a surveillance",
                        "a webcam")):
        return term
    return template.format(term)


def _strip_article(term):
    for art in ("a face ", "a person ", "a photograph of ", "a photo of "):
        if term.startswith(art):
            return term[len(art):]
    return term


def build_expanded_bank(include_race: bool = False, n_templates: int = 3,
                        compositions: bool = True):
    """Return the full deduplicated phrase list, original bank first."""
    phrases = list(_original_bank())
    axes = dict(AXES)

    for axis, terms in axes.items():
        for term in terms:
            for tmpl in TEMPLATES[:max(1, n_templates)]:
                phrases.append(_fmt(tmpl, term))

    if compositions:
        for a, b in COMPOSITIONS:
            if a not in axes or b not in axes:
                continue
            for ta, tb in itertools.product(axes[a], axes[b]):
                head = ta if ta.startswith(("a photo", "a photograph")) else ta
                tail = _strip_article(tb)
                phrases.append(f"{head}, {tail}")

    # appended last so that enabling it never invalidates the cached prefix
    if include_race:
        for term in RACE_TERMS:
            for tmpl in TEMPLATES[:max(1, n_templates)]:
                phrases.append(_fmt(tmpl, term))

    seen, out = set(), []
    for p in phrases:
        key = p.strip().lower().rstrip(".")
        if key and key not in seen:
            seen.add(key)
            out.append(p.strip())
    return out


def bank_sections(include_race: bool = False, n_templates: int = 3,
                  compositions: bool = True):
    """(n_original, n_total) so callers can report how much was appended."""
    original = list(_original_bank())
    full = build_expanded_bank(include_race, n_templates, compositions)
    return len(original), len(full)


if __name__ == "__main__":
    bank = build_expanded_bank()
    n0, n1 = bank_sections()
    print(f"original bank: {n0} phrases")
    print(f"expanded bank: {n1} phrases")
    for p in bank[:5]:
        print("  head :", p)
    for p in bank[n0:n0 + 3]:
        print("  axis :", p)
    for p in bank[-3:]:
        print("  comp :", p)
