"""Slice discovery on Racial Faces in the Wild (RFW) with Domino.

Pipeline stages (see scripts/):
  00  synthetic mini-RFW generator (smoke test)
  01  parse RFW pairs -> manifest
  02  ArcFace (InsightFace) embeddings per image
  03  verification: similarities, global threshold, Platt calibration
  04  CLIP embeddings per image (+ candidate phrase embeddings)
  05  Domino error-aware mixture model over verification pairs
  06  slice report: stats, race composition, montages, text descriptions
"""

__version__ = "0.1.0"

RACES = ["African", "Asian", "Caucasian", "Indian"]
