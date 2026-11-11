"""Evaluate cross-tier confidence calibration.

Question B asks whether the four tiers' confidences are commensurable. This
script quantifies it: we compare RAW confidence vs CALIBRATED P(correct)
against the empirically observed correctness, using

  - ECE (Expected Calibration Error): |confidence - accuracy| averaged over
    confidence bins, weighted by bin population. Lower is better.
  - A per-tier reliability table: for each tier, mean raw conf, mean calibrated
    conf, and the true hit-rate. A well-calibrated score matches the hit-rate.

The held-out lookups are produced exactly as in calibration.py (windowed
holes around each symbol), but here we ALSO run the fold that keeps every
symbol (Tier-1 exact) and, optionally, Tier-4 rows, so the reliability table
spans all tiers on one shared scale.

Run: python3 eval_calibration.py   (offline, no LLM needed for T1-T3)
"""
import os
import numpy as np

from calibration import _load_rows, _kb_dropping_window, _domain_match, KB_PATH
from s7comm_rag.knowledge import KnowledgeBase, Symbol


def collect_eval(rows):
    """Return list of (tier, raw_conf, calibrated, correct) over held-out lookups."""
    gt = {(r.db, r.offset): r.name for r in rows}
    out = []
    # Tier 1: full table, every address hits exactly (correct by construction)
    kb_full = KnowledgeBase()
    for r in rows:
        kb_full.add(Symbol(db=r.db, offset=r.offset, dtype=r.dtype, name=r.name))
    for r in rows:
        res = kb_full.lookup(r.db, r.offset, r.dtype)
        out.append((res.tier, res.confidence, res.calibrated, 1.0))
    # Tiers 2/3: windowed holes of varied radius
    for r in rows:
        for hw in (0, 4, 8, 12, 16):
            kb = _kb_dropping_window(rows, (r.db, r.offset), hw)
            res = kb.lookup(r.db, r.offset, r.dtype)
            if res.tier in (2, 3) and res.symbol is not None:
                correct = 1.0 if _domain_match(res.symbol.name,
                                                gt[(r.db, r.offset)]) else 0.0
                out.append((res.tier, res.confidence, res.calibrated, correct))
    return out


def ece(confs, correct, n_bins=10):
    confs = np.asarray(confs)
    correct = np.asarray(correct)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(confs)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confs[mask].mean()
        err += (mask.sum() / total) * abs(conf - acc)
    return err


def main():
    rows = _load_rows(KB_PATH)
    data = collect_eval(rows)
    tiers = np.array([d[0] for d in data])
    raw = np.array([d[1] for d in data])
    cal = np.array([d[2] for d in data])
    cor = np.array([d[3] for d in data])

    print(f"Held-out lookups: {len(data)}  (from {len(rows)} symbols)\n")

    print("Per-tier reliability (mean raw / mean calibrated / true hit-rate):")
    print(f"{'Tier':<6}{'n':<7}{'raw':<9}{'calibrated':<13}{'hit-rate':<10}")
    print("-" * 45)
    for t in (1, 2, 3, 4):
        m = tiers == t
        if m.sum() == 0:
            continue
        print(f"{t:<6}{int(m.sum()):<7}{raw[m].mean():<9.3f}"
              f"{cal[m].mean():<13.3f}{cor[m].mean():<10.3f}")

    print(f"\nECE (raw confidence)        : {ece(raw, cor):.4f}")
    print(f"ECE (calibrated P(correct)) : {ece(cal, cor):.4f}")
    print("\nLower ECE = confidence better matches real correctness.")
    print("The gap shows how much cross-tier calibration improves reliability.")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
