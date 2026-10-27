"""Natural-language slice descriptions (Domino's *describe* step).

Replicates domino's ``MeanDescriber``: a candidate phrase t scores
``t . (slice_prototype - reference_prototype)`` in CLIP space, where the slice
prototype is the mean image embedding of slice members and the reference is
the overall mean. We additionally report the Pearson correlation between the
phrase-image similarity and slice membership, as the original does.

One adaptation for verification: slicing happens over *pairs*, so the slice
prototype is computed from the CLIP embeddings of the pair's two member
images, weighted by the pair's slice-membership probability. The phrase bank
deliberately covers gender, age, accessories, pose and image quality but not
race/ethnicity terms - the race composition of a slice is computed exactly
from RFW metadata in the report, whereas CLIP race attribution is unreliable
and better avoided.
"""

import numpy as np


def build_phrase_bank():
    """Generates an expanded candidate bank of candidate descriptions for CLIP.
    
    Covers pose, lighting, image quality, age/demographics, facial hair,
    eyewear, occlusions, and facial expressions.
    """
    
    # 1. Pose & Camera View Angle
    poses = [
        "a frontal portrait of a face looking directly at the camera",
        "a side profile view of a face",
        "a face turned at a sharp side angle",
        "a person looking down at the ground",
        "a person looking up at the ceiling",
        "a tilted head photo of a face",
        "an extreme close-up photo of a face",
        "a small face far away from the camera",
    ]

    # 2. Lighting & Shadow Conditions
    lighting = [
        "a dimly lit photo of a face in low light",
        "a photo of a face with severe dark shadows across one side",
        "an overexposed bright photo of a face with flash glare",
        "a backlit photo with bright light behind the person",
        "a high-contrast photo with harsh studio lighting",
        "a photo of a face under dark indoor lighting",
    ]

    # 3. Image Quality, Resolution & Processing
    quality = [
        "a clear high resolution sharp photo of a face",
        "a blurry motion-blurred photo of a face",
        "a grainy low-resolution photo of a face",
        "a pixelated compressed photo of a face",
        "a black and white monochrome photo of a face",
        "an old vintage photograph of a face",
        "a security surveillance camera photo of a face",
    ]

    # 4. Eyewear & Glasses
    glasses = [
        "a person wearing no glasses",
        "a person wearing eyeglasses with clear lenses",
        "a person wearing dark black sunglasses",
        "a person wearing thick-rimmed glasses",
        "a person wearing rimless glasses",
        "a person wearing tinted-lens glasses",
    ]

    # 5. Facial Hair
    facial_hair = [
        "a clean-shaven person with no facial hair",
        "a person with heavy stubble",
        "a person with a full thick beard",
        "a person with a mustache",
        "a person with a goatee",
        "a person with long sideburns",
    ]

    # 6. Age & Life Stage
    age = [
        "a photo of a baby or infant",
        "a photo of a young child",
        "a photo of a teenager",
        "a photo of a young adult",
        "a photo of a middle-aged person",
        "a photo of an elderly senior person with wrinkles",
    ]

    # 7. Headwear, Accessories & Occlusions
    occlusions = [
        "a person wearing a hat or baseball cap",
        "a person wearing a turban, headscarf, or hijab",
        "a person wearing a sports helmet",
        "a face partially covered by a mask or cloth",
        "a face partially hidden by a hand",
        "a person with a microphone in front of their mouth",
        "a person with long hair covering their face",
        "a bald person with no hair",
    ]

    # 8. Expressions & Occlusions
    expressions = [
        "a person with a neutral expression",
        "a smiling person showing teeth",
        "a person laughing with an open mouth",
        "a person shouting or screaming",
        "a person with a serious frowning face",
        "a person crying or looking upset",
    ]

    # Combine all categories
    raw_phrases = (
        poses + lighting + quality + glasses + 
        facial_hair + age + occlusions + expressions
    )

    # Standardize prefix for CLIP text encoder consistency
    formatted_phrases = []
    for p in raw_phrases:
        already = (p.startswith(("a photo", "a face", "a person"))
                   or "photo" in p or "photograph" in p or "portrait" in p)
        if not already:
            p = f"a photo of {p}"
        formatted_phrases.append(p)

    # Deduplicate while preserving order
    return list(dict.fromkeys(formatted_phrases))


def describe_slices(slice_probs: np.ndarray, pair_img_ids: np.ndarray,
                    clip_embs: np.ndarray, text_embs: np.ndarray,
                    phrases, slice_threshold: float = 0.5,
                    n_descriptions: int = 8, top_n_fallback: int = 200,
                    verbose: bool = True):
    """Return, per slice, the top phrases with scores and correlations.

    Args:
        slice_probs: (n_pairs, n_slices) slice membership probabilities.
        pair_img_ids: (n_pairs, 2) image ids of each pair.
        clip_embs: (n_images, d) L2-normalized CLIP image embeddings.
        text_embs: (n_phrases, d) L2-normalized CLIP text embeddings.
    """
    # per-pair image-space embedding = mean of the two member images
    pair_embs = (clip_embs[pair_img_ids[:, 0]] + clip_embs[pair_img_ids[:, 1]]) / 2.0
    ref_proto = pair_embs.mean(axis=0)

    results = []
    for k in range(slice_probs.shape[1]):
        w = slice_probs[:, k]
        mask = w > slice_threshold
        if mask.sum() >= 5:
            slice_proto = pair_embs[mask].mean(axis=0)
        else:
            top = np.argsort(-w)[: max(top_n_fallback, 5)]
            slice_proto = pair_embs[top].mean(axis=0)
        proto_norm = float(np.linalg.norm(slice_proto - ref_proto))

        scores = text_embs @ (slice_proto - ref_proto)
        idxs = np.argsort(-scores)[:n_descriptions]
        phrase_img_scores = text_embs[idxs] @ pair_embs.T  # (n_desc, n_pairs)

        entries = []
        for i, idx in enumerate(idxs):
            x = phrase_img_scores[i]
            corr = float(np.corrcoef(w, x)[0, 1]) if np.std(x) > 0 else 0.0
            entries.append({
                "text": phrases[idx],
                "score": float(scores[idx]),
                "corr": corr,
            })
        if verbose:
            print(f"[describe] slice {k}: {int(mask.sum())} pairs with "
                  f"p>{slice_threshold}, ||proto - ref|| = {proto_norm:.4f}")
        results.append(entries)
    return results
