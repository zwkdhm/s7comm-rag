"""Symbol-table degradation experiment.

Simulates progressive symbol-table loss (new devices added but not registered):
remove entire DBs from the symbol table, then measure semantic recall
WITH vs WITHOUT Tier 4 (vector-KB + LLM inference).

Metric: Semantic Hit Rate = fraction of queries that get a non-trivial
        semantic name (not "unknown_*") that matches the ground-truth domain.
"""
import sys, os, csv, copy
sys.path.insert(0, os.path.dirname(__file__))

from s7comm_rag.knowledge import KnowledgeBase, Symbol
from weak_supervision import (collect_observations, compute_stats,
                              infer_from_statistics)
from generate_real_pcaps import _quantity_family
from eval_weak_supervision import _norm_family
from tier4_llm_inference import VectorKnowledgeBase

KB_PATH = os.environ.get("KB_PATH", "data/mock_symbols.csv")
PCAP    = os.environ.get("TS_PCAP", "data/synthetic_real/timeseries_poll.pcap")
MODEL   = os.environ.get("TIER4_MODEL", "qwen2.5:32b")


def load_ground_truth():
    """Full symbol table = ground truth (kept ONLY for scoring, never fed to
    the Tier-4 inference). Stores the whole row so we can derive the physical-
    quantity family for family-level matching."""
    gt = {}
    with open(KB_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["db"]), int(row["offset"]))
            gt[key] = row
    return gt


def build_kb_with_dropped_dbs(dropped_dbs: set) -> KnowledgeBase:
    """Build a KB that omits all symbols belonging to `dropped_dbs`."""
    kb = KnowledgeBase()
    with open(KB_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["db"]) in dropped_dbs:
                continue
            kb.add(Symbol(
                db=int(row["db"]), offset=int(row["offset"]),
                dtype=row.get("type", ""), name=row["name"],
                unit=row.get("unit", ""),
                range_min=float(row["range_min"]) if row.get("range_min") else None,
                range_max=float(row["range_max"]) if row.get("range_max") else None,
                description=row.get("description", ""),
            ))
    return kb


def run_experiment():
    gt = load_ground_truth()
    all_dbs = sorted({db for (db, _) in gt})

    # Observed value series per point (wire statistics — the ONLY thing Tier 4
    # sees; the symbol table is never fed to inference).
    series = collect_observations([PCAP])
    vkb = VectorKnowledgeBase()

    # Only points that have both a ground-truth row and enough observations.
    queries = sorted(k for (k, dt) in
                     ((( ki[0], ki[1]), ki[2]) for ki in series)
                     if k in gt and len(series[(k[0], k[1], dt)]) >= 5)
    # de-dup (a point may appear once per transport size)
    seen = set(); queries = [k for k in queries if not (k in seen or seen.add(k))]

    sample_n = int(os.environ.get("SAMPLE_N", "0"))
    if sample_n and len(queries) > sample_n:
        step = len(queries) // sample_n
        queries = queries[::step][:sample_n]

    print(f"Model: {MODEL}   Ground-truth symbols: {len(gt)}  across DBs {all_dbs}")
    print(f"Test queries: {len(queries)}   (Tier 4 = value-range inference, no unit)\n")
    print(f"{'Dropped DBs':<14}{'Coverage':<10}{'NoTier4 Hit%':<14}{'+Tier4 Hit%':<13}{'Gain'}")
    print("-" * 60)

    # progressively drop DBs: none, then last-k DBs
    drop_schedule = [set()] + [set(all_dbs[-k:]) for k in range(1, len(all_dbs) + 1)]

    for dropped in drop_schedule:
        kb = build_kb_with_dropped_dbs(dropped)
        coverage = 1 - len(dropped) / len(all_dbs)

        hit_no_t4 = hit_t4 = 0
        for (db, off) in queries:
            dtype = _dtype_of(db, off)
            gt_fam = _quantity_family(gt[(db, off)])
            # WITHOUT Tier 4 (structured/symbol-table tiers only)
            r0 = kb.lookup(db, off, dtype, use_llm=False)
            t13_hit = r0.symbol is not None and _sym_family(r0.symbol) == gt_fam
            if t13_hit:
                hit_no_t4 += 1
            # WITH Tier 4: on a Tier 1-3 miss, infer the family from wire
            # statistics ONLY (value range / magnitude / behaviour), no unit.
            if t13_hit:
                hit_t4 += 1
                continue
            vals = _series_of(series, db, off, dtype)
            if not vals:
                continue
            feats = compute_stats(vals)
            r = infer_from_statistics(feats, vkb, model=MODEL, use_rag=True)
            if _norm_family(r.get("quantity", "")) == gt_fam:
                hit_t4 += 1

        n = len(queries)
        p0 = 100 * hit_no_t4 // n
        p1 = 100 * hit_t4 // n
        dropped_str = "none" if not dropped else ",".join(map(str, sorted(dropped)))
        print(f"{dropped_str:<14}{coverage*100:>5.0f}%    {p0:>8}%      {p1:>8}%     +{p1-p0}%")


def _series_of(series, db, off, dtype):
    """Fetch the observed value list for a point, matching transport size."""
    key = (db, off, dtype)
    if key in series:
        return series[key]
    for (d, o, dt), vals in series.items():
        if d == db and o == off:
            return vals
    return []


def _sym_family(sym) -> str:
    """Physical-quantity family of a symbol RETURNED by Tier 1-3, computed from
    that symbol's own (name, unit)—i.e. the family of whatever entry the tier
    matched. Tier 1 (exact) returns the point itself; Tier 2/3 may return a
    neighbouring/similar entry, which is then judged by ITS true family."""
    return _quantity_family({"name": sym.name, "unit": sym.unit or ""})


# --- helper: read dtype from the ground-truth CSV (structural field only) ----

_DTYPE_CACHE = {}
def _dtype_of(db, off):
    if not _DTYPE_CACHE:
        with open(KB_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _DTYPE_CACHE[(int(row["db"]), int(row["offset"]))] = row.get("type", "")
    return _DTYPE_CACHE.get((db, off), "")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run_experiment()
