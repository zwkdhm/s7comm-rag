"""Generate the ENTITY-relation multi-hop scenario (pcap + ground-truth table).

Counterpart to multihop_generate.py (pointer scenario). Here the decisive clue
is NOT a reference code but ENTITY CONTEXT: two signals are statistically almost
identical (both smooth, normalized 0-100 percent) yet belong to different
physical-quantity families — separable only by WHICH TANK they sit in.

  copper-removal tank  : the normalized signal is an ion CONCENTRATION
  cobalt-removal tank  : the normalized signal is a redox POTENTIAL

This is grounded in real zinc hydrometallurgy (cobalt removal is redox-potential
controlled; cf. voltage tags on cobalt tanks in data/real_symbols.csv). A single
retrieval on the statistics query cannot separate them; only pulling the tank's
role document via the entity graph (same-tank / process-step context) resolves it.

The ground-truth CSV carries a `name` field so entity_graph.EntityGraph can parse
the tank membership (stage/removal/tank), unlike multihop_symbols.csv.
"""
import os, sys, time, random
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from generate_real_pcaps import _s7_write, _write_pcap

OUT_PCAP = "data/synthetic_real/entity_poll.pcap"
OUT_CSV  = "data/entity_symbols.csv"
N_SAMPLES = 60

# (db, offset, name, gt_family, unit, lo, hi, drift_f, noise_f, step_p, step_sz)
# Confusable pairs: same normalized 0-100 % span and smooth behaviour, but the
# tank (encoded in `name`) fixes the family. Copper tank -> concentration,
# cobalt tank -> potential. Statistics are matched across the pair.
ADDRESSES = [
    # --- confusable pair #1 : copper conc vs cobalt potential ---------------
    (1,  0, "purification_1_copper_removal_tank_concentration",
     "concentration", "%", 0, 100, 0.0120, 0.0050, 0.0, 0.0),
    (2,  0, "purification_1_cobalt_removal_tank_potential",
     "potential",     "%", 0, 100, 0.0120, 0.0050, 0.0, 0.0),
    # --- confusable pair #2 -------------------------------------------------
    (1,  4, "purification_2_copper_removal_tank_concentration",
     "concentration", "%", 0, 100, 0.0130, 0.0055, 0.0, 0.0),
    (2,  4, "purification_2_cobalt_removal_tank_potential",
     "potential",     "%", 0, 100, 0.0130, 0.0055, 0.0, 0.0),
    # --- confusable pair #3 -------------------------------------------------
    (1,  8, "purification_3_copper_removal_tank_concentration",
     "concentration", "%", 0, 100, 0.0110, 0.0048, 0.0, 0.0),
    (2,  8, "purification_3_cobalt_removal_tank_potential",
     "potential",     "%", 0, 100, 0.0110, 0.0048, 0.0, 0.0),
    # --- unambiguous controls (stats alone suffice) ------------------------
    (3,  0, "purification_copper_removal_tank_waste_flow_rate",
     "flow", "m3/h", 0, 30, 0.0300, 0.0200, 0.0, 0.0),
    (3,  4, "electrolysis_mix_tank_zinc_feed_rate",
     "feed", "kg/h", 0, 1000, 0.0100, 0.0080, 0.10, 0.15),
]


def _series(off, gt, lo, hi, drift_f, noise_f, step_p, step_sz, n=N_SAMPLES):
    rng = random.Random(hash((off, gt)) & 0xFFFFFFFF)
    span = hi - lo
    op = lo + rng.uniform(0.4, 0.6) * span
    x, vals = op, []
    for _ in range(n):
        x += rng.gauss(0, drift_f * span)
        if step_p and rng.random() < step_p:
            x += rng.choice((-1, 1)) * step_sz * span
        x = max(lo, min(hi, x))
        vals.append(round(max(lo, min(hi, x + rng.gauss(0, noise_f * span))), 2))
    return vals


def main():
    Path("data/synthetic_real").mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("db,offset,type,name,gt_family,unit,range_min,range_max\n")
        for db, off, name, gt, unit, lo, hi, *_ in ADDRESSES:
            f.write(f"{db},{off},REAL,{name},{gt},{unit},{lo},{hi}\n")

    series = {(db, off): _series(off, gt, lo, hi, df, nf, sp, sz)
              for db, off, name, gt, unit, lo, hi, df, nf, sp, sz in ADDRESSES}
    pkts, t = [], time.time()
    for k in range(N_SAMPLES):
        for db, off, *_ in ADDRESSES:
            pkts.append((t, _s7_write(db, off, "REAL", series[(db, off)][k],
                                      ref=k % 60000 + 1)))
            t += 0.05
        t += 1.0
    _write_pcap(OUT_PCAP, pkts)
    print(f"[gen] {OUT_PCAP}  ({len(pkts)} S7 packets, {len(ADDRESSES)} addresses)")
    print(f"[gen] {OUT_CSV}  (ground-truth families + names)")

    from weak_supervision import compute_stats
    print(f"\n{'Address':<10}{'gt_family':<14}{'range':>16}{'cv':>7}{'ac1':>7}")
    print("-" * 54)
    for db, off, name, gt, unit, lo, hi, *_ in ADDRESSES:
        st = compute_stats(series[(db, off)])
        rng = f"[{st['min']:.1f},{st['max']:.1f}]"
        print(f"DB{db}.{off:<6}{gt:<14}{rng:>16}{st['cv']:>7.3f}"
              f"{st.get('autocorr1', 0):>7.3f}")


if __name__ == "__main__":
    main()
