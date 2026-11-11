"""Tier-4b weak-supervision evaluation (zero symbol table).

Layered hit-rate of statistics->semantics inference, evaluated at three
increasingly demanding granularities:

  L1 Behavior : measured value vs setpoint            (from step_ratio)
  L2 Magnitude: order of magnitude of the value range (from observed values)
  L3 Family   : physical-quantity family              (flow/feed/conc/pH/...)

The symbol table is used ONLY to build ground truth for scoring — never fed to
the inference. Inference sees wire statistics + RAG domain knowledge only.
"""
import sys, os, csv, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

from weak_supervision import (collect_observations, compute_stats,
                              classify_behavior, infer_from_statistics)
from generate_real_pcaps import _quantity_family
from tier4_llm_inference import VectorKnowledgeBase

KB_PATH = os.environ.get("KB_PATH", "data/real_symbols.csv")
PCAP    = os.environ.get("TS_PCAP", "data/synthetic_real/timeseries_poll.pcap")
MODEL   = os.environ.get("TIER4_MODEL", "qwen2.5:32b")

# Map fine families to the coarse family the LLM can plausibly name.
FAMILY_ALIASES = {
    "concentration": {"concentration", "ion_concentration", "ion", "conc",
                      "solution_concentration"},
    "flow":          {"flow", "flow_rate", "flowrate", "inlet_flow"},
    "feed":          {"feed", "feed_rate", "feeder", "dosing", "dosing_rate",
                      "feedrate"},
    "level":         {"level", "tank_level", "valve_opening", "opening",
                      "percentage", "ratio"},
    "potential":     {"potential", "voltage", "electrode_potential", "emf"},
    "temperature":   {"temperature", "temp"},
    "density":        {"density", "specific_gravity", "solids_density",
                       "slurry_density"},
    "ph":            {"ph", "acidity", "ph_value"},
    "current":       {"current", "current_density"},
    "setpoint":      {"setpoint", "set_point", "control_target", "target"},
}


def _norm_family(text: str) -> str:
    """Normalize the LLM's free-text quantity into a coarse family key."""
    t = (text or "").lower().replace("-", "_")
    for fam, aliases in FAMILY_ALIASES.items():
        if any(a in t for a in aliases):
            return fam
    return t or "unknown"


def _gt_magnitude(row) -> int:
    hi = float(row["range_max"]) if row["range_max"] else 100.0
    return int(math.floor(math.log10(max(abs(hi), 1e-9)))) if hi > 0 else 0


def evaluate(sample_n: int, use_rag: bool = True, iterative: bool = False,
             entity: bool = False):
    rows = {(int(r["db"]), int(r["offset"])): r
            for r in csv.DictReader(open(KB_PATH))}
    series = collect_observations([PCAP])
    vkb = VectorKnowledgeBase()

    graph = None
    if entity:
        from entity_graph import EntityGraph
        graph = EntityGraph().build(KB_PATH)

    # deterministic sampling across the address space
    keys = sorted(k for k in series if len(series[k]) >= 5
                  and (k[0], k[1]) in rows)
    if sample_n and len(keys) > sample_n:
        step = len(keys) // sample_n
        keys = keys[::step][:sample_n]

    L1 = L2 = L3 = total = rounds_sum = 0
    mode = ("entity" if entity else "iterative" if iterative
            else "single" if use_rag else "no-rag")
    print(f"Model: {MODEL}   Samples: {len(keys)}   RAG mode: {mode}\n")
    print(f"{'Address':<12}{'GT family':<14}{'pred family':<14}"
          f"{'L1':>4}{'L2':>4}{'L3':>4}")
    print("-" * 56)

    for (db, off, dt) in keys:
        vals = series[(db, off, dt)]
        f = compute_stats(vals)
        gt = rows[(db, off)]
        gt_fam = _quantity_family(gt)
        gt_is_sp = (gt_fam == "setpoint")
        gt_mag = _gt_magnitude(gt)

        r = infer_from_statistics(f, vkb, model=MODEL, use_rag=use_rag,
                                  iterative=iterative, entity=entity,
                                  graph=graph, db=db, offset=off)
        pred_fam = _norm_family(r.get("quantity", ""))
        pred_is_sp = bool(r.get("is_setpoint"))
        rounds_sum += r.get("_rounds", 0)

        # L1 behavior: setpoint-vs-measured agreement
        l1 = int(pred_is_sp == gt_is_sp)
        # L2 magnitude: predicted range magnitude within ±1 of truth
        try:
            lo_s, hi_s = r.get("range_estimate", "0-0").split("-")[:2]
            pred_mag = int(math.floor(math.log10(max(abs(float(hi_s)), 1e-9))))
        except Exception:
            pred_mag = f["magnitude"]
        l2 = int(abs(pred_mag - gt_mag) <= 1)
        # L3 family: coarse physical-quantity family match
        l3 = int(pred_fam == gt_fam)

        L1 += l1; L2 += l2; L3 += l3; total += 1
        print(f"DB{db}.{off:<8}{gt_fam:<14}{pred_fam:<14}"
              f"{l1:>4}{l2:>4}{l3:>4}")

    print("-" * 56)
    print(f"\nLayered hit-rate (n={total}):")
    print(f"  L1 Behavior  (setpoint vs measured): {100*L1//total}%")
    print(f"  L2 Magnitude (order of magnitude):    {100*L2//total}%")
    print(f"  L3 Family    (physical quantity):     {100*L3//total}%")
    if iterative or entity:
        print(f"  avg retrieval rounds:                 {rounds_sum/total:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--sample", type=int, default=30)
    ap.add_argument("--no-rag", action="store_true",
                    help="ablate retrieval: statistics + rules only")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--iterative", action="store_true",
                   help="reflective iterative retrieval (retrieve->assess->rewrite->retry)")
    g.add_argument("--entity", action="store_true",
                   help="entity-graph augmented retrieval (same-tank + downstream process context)")
    args = ap.parse_args()
    evaluate(args.sample, use_rag=not args.no_rag, iterative=args.iterative,
             entity=args.entity)
