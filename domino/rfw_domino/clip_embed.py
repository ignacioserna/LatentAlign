"""Stage 04 - cross-modal (CLIP) embeddings for every image, plus embeddings
of a bank of candidate description phrases (used by the describe step).

Domino's key design choice is to slice in a *cross-modal* embedding space
rather than in the task model's own feature space: slices become more coherent
and can be described with natural language. We use OpenAI CLIP weights via
open_clip by default.

The `fake` backend (8x8 RGB pixels) exists for the synthetic smoke test.
"""

import argparse

import numpy as np

from .describe import build_phrase_bank
from .io_utils import Work, l2_normalize, save_json, tqdm


def fake_embed(paths) -> np.ndarray:
    import cv2

    embs = np.zeros((len(paths), 192), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, desc="fake clip embs")):
        img = cv2.imread(p)  # BGR
        if img is None:
            raise FileNotFoundError(p)
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        embs[i] = small.astype(np.float32).ravel()
    embs -= embs.mean(axis=0, keepdims=True)
    return l2_normalize(embs)


def openclip_embed(paths, model_name: str, pretrained: str, device: str,
                   batch_size: int, phrases):
    import torch
    import open_clip
    from PIL import Image

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="clip embs"):
            batch = [
                preprocess(Image.open(p).convert("RGB"))
                for p in paths[start:start + batch_size]
            ]
            x = torch.stack(batch).to(device)
            f = model.encode_image(x)
            feats.append(f.float().cpu().numpy())
    img_embs = l2_normalize(np.concatenate(feats, axis=0).astype(np.float32))

    tokenizer = open_clip.get_tokenizer(model_name)
    txt_feats = []
    with torch.no_grad():
        for start in range(0, len(phrases), 256):
            toks = tokenizer(phrases[start:start + 256]).to(device)
            f = model.encode_text(toks)
            txt_feats.append(f.float().cpu().numpy())
    txt_embs = l2_normalize(np.concatenate(txt_feats, axis=0).astype(np.float32))
    return img_embs, txt_embs, device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work")
    ap.add_argument("--backend", choices=["openclip", "fake"], default="openclip")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai",
                    help="open_clip pretrained tag (openai = original CLIP, "
                    "as used in the Domino paper)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args(argv)

    work = Work(args.work)
    images = work.load_images()
    paths = images["path"].tolist()
    phrases = build_phrase_bank()

    if args.backend == "fake":
        embs = fake_embed(paths)
        meta = {"backend": "fake", "dim": int(embs.shape[1]), "text": False}
    else:
        embs, txt_embs, device = openclip_embed(
            paths, args.model, args.pretrained, args.device, args.batch, phrases
        )
        np.save(work.text_embs_npy, txt_embs)
        with open(work.phrases_txt, "w") as f:
            f.write("\n".join(phrases))
        meta = {"backend": "openclip", "model": args.model,
                "pretrained": args.pretrained, "device": device,
                "dim": int(embs.shape[1]), "text": True,
                "n_phrases": len(phrases)}
        print(f"[clip] wrote {txt_embs.shape} phrase embeddings "
              f"-> {work.text_embs_npy}")

    np.save(work.clip_embs_npy, embs)
    save_json(work.clip_meta_json, meta)
    print(f"[clip] wrote {embs.shape} -> {work.clip_embs_npy}")


if __name__ == "__main__":
    main()
