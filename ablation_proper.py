"""True ablation study: remove innovations one by one.

C1: Tiered retrieval  → ablated by using flat vector search only
C2: Confidence control → ablated by always outputting result (no "unknown")

Metric: Precision@1 and Hallucination rate (returned wrong answer confidently)
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
from s7comm_rag.knowledge import KnowledgeBase, _text, _build_vocab, _ngram_vec
import numpy as np

random.seed(42)

full_kb = KnowledgeBase()
full_kb.load_csv("data/real_symbols.csv")
all_symbols = list(full_kb._table.values())

# Partial KB: remove 40% of symbols to create missing-address queries
random.shuffle(all_symbols)
n_remove = int(len(all_symbols) * 0.4)
known   = all_symbols[n_remove:]   # in KB
unknown = all_symbols[:n_remove]   # removed, used as test queries

partial_kb = KnowledgeBase()
for s in known:
    partial_kb.add(s)

# Rejection thresholds. NOTE: this ablation uses exact-name matching against
# deleted addresses, so almost no query can be "correct" regardless of tier —
# it therefore CANNOT reveal calibration's value (raw vs calibrated rows behave
# identically here). The quantitative calibration evidence lives in
# eval_calibration.py (domain-level hit criterion, ECE 0.1966 -> 0.0001). The
# calibrated row below only confirms that switching to the calibrated threshold
# does not break the existing confidence-control (rejection) behavior.
CONF_THRESHOLD = 0.6   # raw-confidence rejection threshold
CAL_THRESHOLD  = 0.6   # calibrated P(correct) rejection threshold

# ---- C1 ablation: flat vector search only (no tiered fallback)
def flat_vector_search(kb, db, offset, dtype):
    """No exact/range lookup. Jump straight to vector search for all queries."""
    kb._ensure_embeddings()
    query = _text(type('S', (), {'db': db, 'offset': offset, 'dtype': dtype,
                                  'name': '', 'description': '', 'unit': ''})())
    q_vec = _ngram_vec(query, kb._vocab)
    norm_e = np.linalg.norm(kb._embeddings, axis=1)
    norm_q = np.linalg.norm(q_vec)
    sims = kb._embeddings @ q_vec / (norm_e * norm_q + 1e-9)
    best = int(np.argmax(sims))
    return kb._embed_list[best], float(sims[best])

# ---- run experiment
configs = [
    "完整系统 (Tiered + raw置信度阈值)",
    "完整系统 (Tiered + 校准置信度阈值)",
    "去掉C1 (仅向量检索，无分层)",
    "去掉C2 (分层保留，无置信度过滤)",
]

print(f"已知符号: {len(known)}  测试查询(已删除): {len(unknown)}\n")
print(f"{'配置':35s}  {'Prec@1':>8}  {'有效率':>8}  {'幻觉率':>8}  {'拒答率':>8}")
print("-" * 80)

for cfg in configs:
    correct = answered = hallucinated = refused = 0
    for s in unknown:
        gt = s.name
        if cfg == "完整系统 (Tiered + raw置信度阈值)":
            r = partial_kb.lookup(s.db, s.offset, s.dtype)
            if r.confidence >= CONF_THRESHOLD:
                answered += 1
                if r.symbol and r.symbol.name == gt: correct += 1
                else: hallucinated += 1
            else:
                refused += 1
        elif cfg == "完整系统 (Tiered + 校准置信度阈值)":
            r = partial_kb.lookup(s.db, s.offset, s.dtype)
            if r.calibrated >= CAL_THRESHOLD:
                answered += 1
                if r.symbol and r.symbol.name == gt: correct += 1
                else: hallucinated += 1
            else:
                refused += 1
        elif cfg == "去掉C1 (仅向量检索，无分层)":
            sym, conf = flat_vector_search(partial_kb, s.db, s.offset, s.dtype)
            answered += 1
            if sym.name == gt: correct += 1
            else: hallucinated += 1
        else:  # 去掉C2: tiered but no confidence filter
            r = partial_kb.lookup(s.db, s.offset, s.dtype)
            if r.symbol:
                answered += 1
                if r.symbol.name == gt: correct += 1
                else: hallucinated += 1
            else:
                refused += 1

    total = len(unknown)
    prec      = correct / answered if answered else float('nan')  # 在有回答的里面有多少对
    useful    = correct / total                                    # 对全部查询的贡献
    hall_rate = hallucinated / answered if answered else 0         # 在有回答的里面有多少错
    ref_rate  = refused / total
    print(f"{cfg:35s}  {prec:>10.0%}  {useful:>9.0%}  {hall_rate:>10.0%}  {ref_rate:>6.0%}")
