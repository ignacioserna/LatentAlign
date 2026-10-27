"""Face-attribute vocabulary for describing discovered slices.

Two blocks:

  MAAD_FACE     the 47 attributes of MAAD-Face (Terhoerst et al., 2021), which is
                the usual ground-truth attribute source for face-recognition bias
                studies.  Keeping the vocabulary aligned with MAAD-Face means the
                terms a method *proposes* live in the same space as the terms you
                *evaluate* against, so Precision@k is not confounded by vocabulary
                mismatch.

  COVARIATE     acquisition / quality factors that dominate FR error rates but are
                absent from attribute datasets: pose, blur, illumination, occlusion,
                resolution, expression intensity.  Without these the baselines are
                forced to explain quality-driven failures with identity attributes,
                which makes them look worse than they are.  Include them.

Each term carries a set of prompt templates.  CLIP is sensitive to prompt wording,
so we average the text embedding over templates (the standard "prompt ensembling"
trick) rather than trusting a single phrasing.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# MAAD-Face's 47 attributes, phrased as natural-language noun/adjective terms.
# --------------------------------------------------------------------------
MAAD_FACE = [
    # demographics
    "a man", "a woman", "a young person", "a middle-aged person", "an elderly person",
    "an Asian person", "a White person", "a Black person",
    # skin / face shape
    "rosy cheeks", "shiny skin", "an oval face", "a square face", "a round face",
    "a double chin", "high cheekbones", "a chubby face",
    # hair
    "a bald head", "wavy hair", "a receding hairline", "bangs", "sideburns",
    "black hair", "blond hair", "brown hair", "gray hair",
    # facial hair
    "no beard", "a mustache", "a five o'clock shadow", "a goatee",
    # forehead
    "an obstructed forehead", "a fully visible forehead",
    # eyes / brows
    "brown eyes", "bags under the eyes", "bushy eyebrows", "arched eyebrows",
    # mouth / nose
    "a closed mouth", "a smile", "big lips", "a big nose", "a pointy nose",
    # accessories / styling
    "heavy makeup", "a hat", "earrings", "a necktie", "lipstick",
    "no eyewear", "eyeglasses",
    # subjective
    "an attractive face",
]

# --------------------------------------------------------------------------
# Acquisition covariates.  These are what actually drives FNMR in the wild.
# --------------------------------------------------------------------------
COVARIATE = [
    # pose
    "a frontal view of a face", "a profile view of a face",
    "a face turned to the side", "a face looking down", "a face looking up",
    "a tilted head",
    # quality
    "a blurry photo", "a sharp photo", "a low-resolution photo",
    "a grainy noisy photo", "an overexposed photo", "an underexposed photo",
    "a dark photo", "a brightly lit photo", "harsh directional lighting",
    "a shadow across the face", "a backlit face",
    # occlusion
    "a face partly occluded", "a face behind a microphone",
    "a hand covering part of the face", "sunglasses", "a face mask",
    "a scarf covering the chin", "hair covering the face",
    # capture context
    "a black and white photo", "a photo with a colour cast",
    "a photo taken from far away", "a close-up portrait",
    "a crowded background", "a plain background",
    "a photo taken at night", "an indoor photo", "an outdoor photo",
    "a press photograph", "a webcam photo", "a scanned photo",
    # expression / motion
    "an open mouth talking", "an exaggerated expression",
    "eyes closed", "a face in motion",
]

DEFAULT_VOCAB = MAAD_FACE + COVARIATE

# --------------------------------------------------------------------------
# Prompt templates.  {} is filled with the term.
# --------------------------------------------------------------------------
TEMPLATES = [
    "a photo of {}.",
    "a cropped photo of {}.",
    "a photograph of a person, {}.",
    "{}.",
]


def build_prompts(terms=None, templates=None):
    """Return (terms, prompts, group_index).

    ``prompts`` is flat (len(terms) * len(templates)); ``group_index[i]`` gives the
    term index for prompt i, so you can mean-pool text embeddings per term.
    """
    terms = list(DEFAULT_VOCAB if terms is None else terms)
    templates = list(TEMPLATES if templates is None else templates)
    prompts, group = [], []
    for ti, t in enumerate(terms):
        for tmpl in templates:
            prompts.append(tmpl.format(t))
            group.append(ti)
    return terms, prompts, group


def load_terms(path):
    """Read a newline-delimited vocabulary file, ignoring blanks and # comments."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out
