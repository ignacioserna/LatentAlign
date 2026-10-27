"""Stage 04b - re-encode the description bank with many more prompts.

Stage 04 encodes `describe.build_phrase_bank()` (~53 phrases) alongside the image
embeddings. Slice *description* survives on 53 phrases; slice *discovery* does not.
B2T ranks phrases by a CLIP-score gap and Domino's MeanDescriber can only return a
phrase that exists in the bank, so the bank's granularity is a hard ceiling on both.
`phrase_bank.build_expanded_bank()` raises it to ~1,700.

Only text is re-encoded here. Image embeddings are untouched, so this costs seconds
on CPU and does not invalidate `clip_embs.npy` or anything downstream of it. The
original bank stays at the head of the list at the same indices, so any phrase index
already quoted in a report still resolves to the same phrase.

    python -m rfw_domino.expand_phrases --work work

Re-running is cheap and idempotent: the existing `text_embs.npy` prefix is reused
whenever `phrases.txt` is still a prefix of the new bank, and only the tail is
encoded.

After this, re-run stage 06 (report) to get descriptions from the larger bank, and
run the baselines, which read `text_embs.npy` / `phrases.txt` directly.
"""

import argparse
import json
import os
import shutil

import numpy as np

from .io_utils import Work, l2_normalize
from .phrase_bank import build_expanded_bank


def encode_phrases(phrases, model_name, pretrained, device, batch=256):
    import torch
    import open_clip

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    feats = []
    with torch.no_grad():
        for start in range(0, len(phrases), batch):
            toks = tokenizer(phrases[start:start + batch]).to(device)
            f = model.encode_text(toks)
            feats.append(f.float().cpu().numpy())
            print(f"\r[phrases] encoded {min(start + batch, len(phrases))}"
                  f"/{len(phrases)}", end="", flush=True)
    print()
    return l2_normalize(np.concatenate(feats, axis=0).astype(np.float32)), device


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--model", default=None,
                    help="default: whatever clip_meta.json recorded for the "
                         "image embeddings (they must match)")
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--include-race", action="store_true",
                    help="add race/ethnicity phrases. Off by default, following "
                         "describe.py: RFW gives exact race metadata and CLIP race "
                         "attribution is unreliable. Use only for an ablation.")
    ap.add_argument("--n-templates", type=int, default=3)
    ap.add_argument("--no-compositions", dest="compositions",
                    action="store_false",
                    help="skip two-attribute phrases like 'a blurry photo of an "
                         "elderly person' (these are where the failure modes are)")
    ap.set_defaults(compositions=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the bank size and a sample, encode nothing")
    args = ap.parse_args(argv)

    work = Work(args.work)
    phrases = build_expanded_bank(args.include_race, args.n_templates,
                                  args.compositions)
    print(f"[phrases] expanded bank: {len(phrases)} phrases")

    if args.dry_run:
        for p in phrases[:3] + ["..."] + phrases[-3:]:
            print("   ", p)
        return

    model_name, pretrained = args.model, args.pretrained
    if model_name is None or pretrained is None:
        if not os.path.exists(work.clip_meta_json):
            raise SystemExit(
                "clip_meta.json not found - run stage 04 first, or pass "
                "--model/--pretrained explicitly.")
        with open(work.clip_meta_json) as f:
            meta = json.load(f)
        if meta.get("backend") != "openclip":
            raise SystemExit(
                f"clip_meta.json records backend={meta.get('backend')!r}; text "
                f"embeddings only exist for the openclip backend.")
        model_name = model_name or meta["model"]
        pretrained = pretrained or meta["pretrained"]
    print(f"[phrases] model {model_name}/{pretrained} (matching clip_embs.npy)")

    # reuse the cached prefix if the old bank is still a prefix of the new one
    old_phrases, old_embs = [], None
    if os.path.exists(work.phrases_txt) and os.path.exists(work.text_embs_npy):
        with open(work.phrases_txt) as f:
            old_phrases = [ln.strip() for ln in f if ln.strip()]
        old_embs = np.load(work.text_embs_npy)
        if len(old_phrases) != len(old_embs):
            print("[phrases] cached phrases.txt and text_embs.npy disagree in "
                  "length; re-encoding everything")
            old_phrases, old_embs = [], None

    reuse = 0
    if old_embs is not None and len(old_phrases) <= len(phrases):
        if phrases[:len(old_phrases)] == old_phrases:
            reuse = len(old_phrases)
            print(f"[phrases] reusing {reuse} cached embeddings, encoding "
                  f"{len(phrases) - reuse} new")

    if reuse == len(phrases):
        print("[phrases] nothing to do")
        return

    new_embs, device = encode_phrases(phrases[reuse:], model_name, pretrained,
                                      args.device, args.batch)
    embs = new_embs if reuse == 0 else np.concatenate([old_embs[:reuse], new_embs])

    if old_embs is not None:
        for src, suffix in ((work.text_embs_npy, ".bak.npy"),
                            (work.phrases_txt, ".bak.txt")):
            bak = src + suffix
            if not os.path.exists(bak):
                shutil.copy2(src, bak)
        print(f"[phrases] backed up the previous bank to *.bak.*")

    np.save(work.text_embs_npy, embs)
    with open(work.phrases_txt, "w") as f:
        f.write("\n".join(phrases))

    meta_path = work.path("phrase_bank_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"n_phrases": len(phrases), "model": model_name,
                   "pretrained": pretrained, "device": device,
                   "include_race": args.include_race,
                   "n_templates": args.n_templates,
                   "compositions": args.compositions,
                   "n_original_bank": reuse or None}, f, indent=2)
    print(f"[phrases] wrote {embs.shape} -> {work.text_embs_npy}")
    print(f"[phrases] wrote {len(phrases)} phrases -> {work.phrases_txt}")


if __name__ == "__main__":
    main()
