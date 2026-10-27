"""Face alignment helpers (ArcFace 112x112 five-point template) and parsing of
RFW's `<Race>_lmk.txt` landmark files."""

import numpy as np

# InsightFace's canonical ArcFace destination landmarks for 112x112 crops:
# left eye, right eye, nose, left mouth corner, right mouth corner.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)


def norm_crop(img_bgr: np.ndarray, lmk5: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-transform crop given 5 landmarks (order: leye, reye, nose,
    lmouth, rmouth), as in insightface.utils.face_align.norm_crop."""
    import cv2

    dst = ARCFACE_DST * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(
        lmk5.astype(np.float32), dst, method=cv2.LMEDS
    )
    if M is None:  # degenerate landmarks - fall back to a plain resize
        return cv2.resize(img_bgr, (size, size))
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)


def interpret_landmarks(vals) -> np.ndarray:
    """Turn 10 floats from an RFW lmk line into a (5, 2) array.

    The two common serializations are interleaved ``x1 y1 x2 y2 ...`` and
    MTCNN-style blocked ``x1..x5 y1..y5``. We try both and keep the one that is
    geometrically plausible (eyes above nose above mouth, eyes roughly level).
    """
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size != 10:
        raise ValueError(f"expected 10 landmark values, got {vals.size}")
    interleaved = vals.reshape(5, 2)
    blocked = np.stack([vals[:5], vals[5:]], axis=1)

    def plausibility(p):
        eyes_y = p[:2, 1].mean()
        nose_y = p[2, 1]
        mouth_y = p[3:, 1].mean()
        score = 0.0
        score += 1.0 if eyes_y < nose_y < mouth_y else 0.0
        score += 1.0 if abs(p[0, 1] - p[1, 1]) < abs(p[0, 0] - p[1, 0]) else 0.0
        score += 1.0 if p[0, 0] < p[1, 0] else 0.0  # left eye left of right eye
        return score

    return interleaved if plausibility(interleaved) >= plausibility(blocked) else blocked


def load_lmk_file(path: str) -> dict:
    """Parse an RFW landmark file into {image filename: (5, 2) landmarks}.

    Lines look like ``<path or filename> f1 f2 ... f10`` (occasionally with an
    extra leading field); we take the last 10 floats on the line.
    """
    import os

    lmk = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 11:
                continue
            floats = []
            for t in reversed(toks):
                try:
                    floats.append(float(t))
                except ValueError:
                    break
                if len(floats) == 10:
                    break
            if len(floats) != 10:
                continue
            floats = floats[::-1]
            key = os.path.basename(toks[0])
            lmk[key] = interpret_landmarks(floats)
    return lmk
