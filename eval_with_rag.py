"""RAG comparison experiment.

Four-way evaluation on synthetic pcaps (which have full symbol-table coverage):
  A) GPT-2 from raw hex+boundary+endpoint  — end-to-end neural, no symbol info
  B) Parser + Template                     — deterministic structural, no symbol info
  C) Parser + RAG                          — deterministic + symbol semantics (proposed)
  D) GPT-2(enriched) + RAG                 — GPT-2 prediction used as KB query key

Ground truth: semantic description from parser + symbol table (Tier-1 exact match).
Metrics: Field Exact Match (FEM) per field: operation / tag / value / unit.
Reported as FEM_macro (field-macro average of the four per-field rates) and
FEM_joint (fraction of samples with all four fields simultaneously correct).
Two scoring conventions matter when reading the value/unit columns (see
fem_score): a ground truth carrying no value= counts as correct for any
prediction, and a ground truth carrying no unit= counts as correct only when
the prediction also omits unit=.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

from s7comm_parser import parse_pcap, _s7_offset
from s7comm_rag.knowledge import KnowledgeBase
from generate_training_pairs import build_input
from scapy.all import rdpcap, TCP, Raw, IP

SYNTHETIC_PCAPS = os.environ.get(
    "PCAP_LIST",
    "data/synthetic/boiler_startup.pcap,"
    "data/synthetic/pressure_control.pcap,"
    "data/synthetic/fault_scenario.pcap,"
    "data/synthetic/mixed_readwrite.pcap"
).split(",")
KB_PATH    = os.environ.get("KB_PATH", "data/mock_symbols.csv")
GPT2_MODEL = os.environ.get("GPT2_MODEL", "data/s7comm-gpt2")


# ----------------------------------------------------------------- generation methods

def method_template(ev, item, val) -> str:
    """B: deterministic parser, no symbol table."""
    addr = f"DB{item.db_number} offset {item.byte_offset} type {item.transport_size}"
    if ev.function == "Write" and val is not None:
        return f"Write {addr} value={val}"
    return f"Read {addr}"


def method_rag(ev, item, val, kb: KnowledgeBase) -> str:
    """C: deterministic parser + RAG symbol lookup."""
    r = kb.lookup(item.db_number, item.byte_offset, item.transport_size)
    if not r.symbol:
        return method_template(ev, item, val)
    name = r.symbol.description or r.symbol.name
    unit = r.symbol.unit or ""
    if ev.function == "Write" and val is not None:
        return f"Write {name} value={val}{unit}"
    return f"Read {name}{' unit=' + unit if unit else ''}"


def method_gpt2_rag(gpt2_pred: str, kb: KnowledgeBase) -> str:
    """D: parse GPT-2's structural output, then query KB with predicted address.
    RAG quality is bounded by GPT-2 structural accuracy (~29%).
    """
    m = re.search(r'DB(\d+)\s+offset\s+(\d+)\s+type\s+(\S+)', gpt2_pred, re.IGNORECASE)
    if not m:
        return gpt2_pred
    db, offset, dtype = int(m.group(1)), int(m.group(2)), m.group(3)
    val_m = re.search(r'value=(\S+)', gpt2_pred)
    val   = val_m.group(1) if val_m else None

    r = kb.lookup(db, offset, dtype)
    if not r.symbol:
        return gpt2_pred
    name = r.symbol.description or r.symbol.name
    unit = r.symbol.unit or ""
    op = gpt2_pred.split()[0].capitalize()
    if op == "Write" and val is not None:
        return f"Write {name} value={val}{unit}"
    return f"Read {name}{' unit=' + unit if unit else ''}"


def ground_truth(ev, item, val, kb: KnowledgeBase):
    """Exact semantic description using Tier-1 KB match only."""
    r = kb.lookup(item.db_number, item.byte_offset, item.transport_size)
    if not r.symbol or r.tier != 1:
        return None
    name = r.symbol.description or r.symbol.name
    unit = r.symbol.unit or ""
    if ev.function == "Write" and val is not None:
        return f"Write {name} value={val}{unit}"
    return f"Read {name}{' unit=' + unit if unit else ''}"


# ----------------------------------------------------------------- FEM scoring

def fem_score(predicted: str, truth: str) -> dict:
    """Per-field exact match. Two conventions the caller must report:
      - value: if the truth has no value= (a Read event), the field is vacuously
        correct regardless of the prediction.
      - unit:  if the truth has no unit= (a Write event, where the unit is
        appended to the value string), the field is correct only when the
        prediction also omits unit=, i.e. both sides compare as "".
      - tag:   the whole token span after the operation and before value=, so it
        jointly covers the semantic label plus any type/unit remainder.
    """
    if not predicted or not truth:
        return {"operation": 0, "tag": 0, "value": 0, "unit": 0}
    pred_l, true_l = predicted.lower(), truth.lower()

    op_p = pred_l.split()[0]
    op_t = true_l.split()[0]

    val_p = (re.search(r'value=([\S]+)', pred_l) or [None, ""])[1]
    val_t = (re.search(r'value=([\S]+)', true_l) or [None, ""])[1]

    unit_p = (re.search(r'unit=(\S+)', pred_l) or [None, ""])[1]
    unit_t = (re.search(r'unit=(\S+)', true_l) or [None, ""])[1]

    tag_p = pred_l.split("value=")[0].split()[1:] if "value=" in pred_l else pred_l.split()[1:]
    tag_t = true_l.split("value=")[0].split()[1:] if "value=" in true_l else true_l.split()[1:]

    return {
        "operation": int(op_p == op_t),
        "tag":       int(" ".join(tag_p) == " ".join(tag_t)),
        "value":     int(val_p == val_t) if val_t else 1,
        "unit":      int(unit_p == unit_t),
    }


# ----------------------------------------------------------------- collect samples

def collect(kb: KnowledgeBase) -> list:
    samples = []
    for path in SYNTHETIC_PCAPS:
        events = parse_pcap(path)
        pkts   = rdpcap(path)
        s7_items = []
        for pkt in pkts:
            if not (pkt.haslayer(TCP) and pkt.haslayer(Raw)):
                continue
            data = bytes(pkt[Raw])
            off  = _s7_offset(data)
            if off and len(data) > off + 10 and data[off + 10] in (0x04, 0x05):
                s7_items.append((data[off:], pkt))

        for ev, (s7, pkt) in zip(events, s7_items):
            if not ev.items:
                continue
            item = ev.items[0]
            val  = ev.values[0] if ev.values else None
            gt   = ground_truth(ev, item, val, kb)
            if gt is None:
                continue
            samples.append({
                "gt":       gt,
                "template": method_template(ev, item, val),
                "rag":      method_rag(ev, item, val, kb),
                "gpt2_in":  build_input(s7, pkt),
            })
    return samples


# ----------------------------------------------------------------- main

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.load_csv(KB_PATH)
    samples = collect(kb)
    print(f"Samples with ground truth: {len(samples)}\n")

    # GPT-2 methods only if model exists (skip for real-data runs with no trained model)
    methods = {
        "B. Parser+Template":             [s["template"] for s in samples],
        "C. Parser+RAG (proposed)":       [s["rag"]      for s in samples],
    }
    if os.path.exists(GPT2_MODEL):
        from finetune_gpt2 import generate
        gpt2_preds = [generate(GPT2_MODEL, s["gpt2_in"]).strip() for s in samples]
        methods["A. GPT-2 (enriched)"] = gpt2_preds
        methods["D. GPT-2(enriched)+RAG"] = [method_gpt2_rag(p, kb) for p in gpt2_preds]

    n_no_val  = sum(1 for s in samples if "value=" not in s["gt"].lower())
    n_no_unit = sum(1 for s in samples if "unit="  not in s["gt"].lower())
    print(f"Ground truths without a value= field: {n_no_val}/{len(samples)} "
          f"(scored as correct by convention)")
    print(f"Ground truths without a unit= field:  {n_no_unit}/{len(samples)} "
          f"(scored as correct only if the prediction also omits it)\n")

    fields = ("operation", "tag", "value", "unit")
    print(f"{'Method':<30} {'Op%':>6} {'Tag%':>6} {'Val%':>6} {'Unit%':>6} "
          f"{'Macro%':>7} {'Joint%':>7}")
    print("-" * 78)
    for name, preds in methods.items():
        scores = [fem_score(p, s["gt"]) for p, s in zip(preds, samples)]
        n = len(scores)
        per_field = {f: 100.0 * sum(s[f] for s in scores) / n for f in fields}
        macro = sum(per_field.values()) / 4
        joint = 100.0 * sum(1 for s in scores if all(s[f] for f in fields)) / n
        print(f"{name:<30}" + "".join(f"{per_field[f]:>5.1f}% " for f in fields)
              + f"{macro:>6.2f}% {joint:>6.2f}%")
