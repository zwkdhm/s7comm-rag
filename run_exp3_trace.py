"""Exp. 3 — Tier-4 inference-chain trace for a single held-out point (DB1.20).

Zero-leakage chain (same protocol as eval_degradation.py / §D of the paper):
  wire value series -> compute_stats -> infer_from_statistics
The query is built from wire-observable statistics only (value range, magnitude,
temporal shape); no unit or symbol name enters the inference. REAL is an IEEE-754
float and carries no engineering unit, so the unit is NOT recoverable from the wire.

Ground truth: purification_1_copper_removal_tank_concentration (copper-ion conc).
Expected outcome: with only the small 4-5 magnitude range and no unit, the category
is intrinsically ambiguous between concentration (g/L) and electrode potential (V),
so Tier 4 misjudges it as electrode_potential — the direct cause of the 40% ceiling.
Model set via TIER4_MODEL.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tier4_llm_inference import VectorKnowledgeBase
from weak_supervision import (collect_observations, compute_stats,
                              build_stat_query, infer_from_statistics)

MODEL = os.environ.get("TIER4_MODEL", "qwen2.5:32b")
PCAP  = os.environ.get("TS_PCAP", "data/synthetic_real/timeseries_poll.pcap")
DB, OFFSET, DTYPE = 1, 20, "REAL"


def _series_of(series, db, off, dtype):
    key = (db, off, dtype)
    if key in series:
        return series[key]
    for (d, o, _dt), vals in series.items():
        if d == db and o == off:
            return vals
    return []


vkb = VectorKnowledgeBase()
series = collect_observations([PCAP])
vals = _series_of(series, DB, OFFSET, DTYPE)

print(f"Model: {MODEL}")
print(f"\n[1] deterministic parse: DB={DB}, offset={OFFSET}, type={DTYPE} "
      f"(REAL carries no unit)")
print(f"    {len(vals)} observations, "
      f"range {min(vals):.2f}-{max(vals):.2f}" if vals else "    (no observations)")

feats = compute_stats(vals)
query = build_stat_query(feats)
print(f"\n[3] leakage-free query (statistics only, no unit/name):\n  {query}")

r = infer_from_statistics(feats, vkb, model=MODEL, use_rag=True)
print(f"\n[5] LLM inference ({MODEL}):")
print("  " + json.dumps(r, ensure_ascii=False))
print(f"\n[6] ground truth: purification_1_copper_removal_tank_concentration "
      f"(copper-ion concentration)")
