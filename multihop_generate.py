"""Generate the multi-hop disambiguation scenario (pcap + ground-truth table).

Motivation
----------
In the real zero-symbol-table data the wire statistics are (nearly) self-
sufficient: sign + magnitude + temporal behaviour already pin the physical-
quantity family, so a single retrieval round is enough and reflective iterative
retrieval degrades to single-round (no gain).

This scenario is deliberately constructed as a CASE STUDY of *when* iterative
retrieval is worth its cost: a set of addresses whose statistical signatures are
mutually AMBIGUOUS (all 0-100, non-negative, smooth), so neither the prompt
rules nor a single corpus lookup can separate them. The decisive knowledge is
laid out as a TWO-HOP chain in the corpus (see multihop_build_kb.py):

  hop-1  a "bridge" note reachable from the statistics query — it only says the
         0-100 signals in the electrowinning stage are ambiguous and points to a
         separate variable-role reference; it does NOT contain the answer.
  hop-2  the variable-role reference (reachable only after the query is rewritten
         with the bridge's pointer) — it maps the fine statistical signature
         (very low cv + high autocorrelation => normalized cell potential;
         freer slow drift => tank level) to the family.

Ground truth families are stored EXPLICITLY here (not derived from unit), because
the ambiguity is the whole point: two families share the unit "%".
"""
import os, sys, time, random, math
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from generate_real_pcaps import _s7_write, _write_pcap

OUT_PCAP = "data/synthetic_real/multihop_poll.pcap"
OUT_CSV  = "data/multihop_symbols.csv"
N_SAMPLES = 60

# (db, offset, gt_family, unit, lo, hi, drift_frac, noise_frac, step_p, step_sz)
# Ambiguous pairs share unit "%" and the 0-100 range; only the fine temporal
# signature (cv / autocorr) separates them, and that mapping lives in hop-2.
ADDRESSES = [
    # --- ambiguous 0-100 pair #1 -------------------------------------------
    # tightly rectifier-regulated => very low cv, high autocorr => POTENTIAL
    (1,   0, "potential", "%", 40, 60, 0.0030, 0.0015, 0.00, 0.0),
    # freely integrating tank level => higher drift/cv => LEVEL
    (1,   4, "level",     "%",  0, 100, 0.0220, 0.0060, 0.00, 0.0),
    # --- ambiguous 0-100 pair #2 -------------------------------------------
    (1,   8, "potential", "%", 30, 70, 0.0032, 0.0016, 0.00, 0.0),
    (1,  12, "level",     "%",  0, 100, 0.0200, 0.0055, 0.00, 0.0),
    # --- ambiguous 0-100 pair #3 -------------------------------------------
    (1,  16, "potential", "%", 45, 65, 0.0028, 0.0014, 0.00, 0.0),
    (1,  20, "level",     "%",  0, 100, 0.0240, 0.0065, 0.00, 0.0),
    # --- unambiguous controls (both methods should get these) --------------
    (2,   0, "concentration", "g/L",   0, 200, 0.0100, 0.0040, 0.00, 0.0),
    (2,   4, "flow",          "m3/h",  0,  30, 0.0300, 0.0200, 0.00, 0.0),
    (2,   8, "potential",     "mV", -700,   0, 0.0120, 0.0060, 0.02, 0.05),
    (2,  12, "feed",          "kg/h",  0, 1000, 0.0100, 0.0080, 0.10, 0.15),
    # --- chain B: Modbus skid (DB3), 0-100% ambiguous density vs flow -------
    # hop-2 Table 7-3 rule: slow integrating ramp + high ac1 => DENSITY;
    #                       fast mean-reverting oscillation    => FLOW
    (3,   0, "density", "%",  0, 100, 0.0200, 0.0050, 0.00, 0.0),
    (3,   4, "flow",    "%",  0, 100, 0.0050, 0.0230, 0.00, 0.0),
    (3,   8, "density", "%",  0, 100, 0.0210, 0.0055, 0.00, 0.0),
    (3,  12, "flow",    "%",  0, 100, 0.0055, 0.0240, 0.00, 0.0),
    # --- chain C: P&ID leaching loop (DB4), 0-100% ambiguous temp vs level --
    # hop-2 FT-2201 rule: cv<0.05 + near-unity ac1 (cascade-pinned) => TEMPERATURE;
    #                     larger-variance slow drift                => LEVEL
    (4,   0, "temperature", "%", 40, 60,  0.0030, 0.0015, 0.00, 0.0),
    (4,   4, "level",       "%",  0, 100, 0.0220, 0.0060, 0.00, 0.0),
    (4,   8, "temperature", "%", 35, 65,  0.0032, 0.0016, 0.00, 0.0),
    (4,  12, "level",       "%",  0, 100, 0.0200, 0.0055, 0.00, 0.0),
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

    # ground-truth symbol table (families explicit; used ONLY for scoring)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("db,offset,type,gt_family,unit,range_min,range_max\n")
        for db, off, gt, unit, lo, hi, *_ in ADDRESSES:
            f.write(f"{db},{off},REAL,{gt},{unit},{lo},{hi}\n")

    # time-series pcap (cyclic polling, one Write per observed value)
    series = {(db, off): _series(off, gt, lo, hi, df, nf, sp, sz)
              for db, off, gt, unit, lo, hi, df, nf, sp, sz in ADDRESSES}
    pkts, t = [], time.time()
    for k in range(N_SAMPLES):
        for db, off, *_ in ADDRESSES:
            pkts.append((t, _s7_write(db, off, "REAL", series[(db, off)][k],
                                      ref=k % 60000 + 1)))
            t += 0.05
        t += 1.0
    _write_pcap(OUT_PCAP, pkts)
    print(f"[gen] {OUT_PCAP}  ({len(pkts)} S7 packets, {len(ADDRESSES)} addresses)")
    print(f"[gen] {OUT_CSV}  (ground-truth families)")

    # report the statistical signatures so the ambiguity is visible
    from weak_supervision import compute_stats
    print(f"\n{'Address':<10}{'gt_family':<14}{'range':>16}{'cv':>7}{'ac1':>7}")
    print("-" * 54)
    for db, off, gt, unit, lo, hi, *_ in ADDRESSES:
        st = compute_stats(series[(db, off)])
        rng = f"[{st['min']:.1f},{st['max']:.1f}]"
        print(f"DB{db}.{off:<6}{gt:<14}{rng:>16}{st['cv']:>7.3f}"
              f"{st.get('autocorr1', 0):>7.3f}")


if __name__ == "__main__":
    main()
