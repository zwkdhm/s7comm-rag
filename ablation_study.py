"""Ablation study: compare retrieval precision under partial symbol tables.

For each symbol-table coverage level (10%~100%) and each tier configuration,
measure Precision@1 (did the top result match ground truth?) and
Coverage (fraction of queries that returned any result).
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
from s7comm_rag.knowledge import KnowledgeBase, Symbol, RANGE_THRESHOLD

random.seed(42)

# ---------------------------------------------------------------- load full KB
full_kb = KnowledgeBase()
full_kb.load_csv("data/mock_symbols.csv")
all_symbols = list(full_kb._table.values())

# ---------------------------------------------------------------- tier configs
def lookup_tier1_only(kb, db, offset, dtype):
    """Only exact match; no fallback."""
    sym = kb._table.get((db, offset))
    return sym

def lookup_tier12(kb, db, offset, dtype):
    """Exact + range, no semantic."""
    sym = kb._table.get((db, offset))
    if sym:
        return sym
    if kb._by_db.get(db):
        nearest = min(kb._by_db[db], key=lambda s: abs(s.offset - offset))
        if abs(nearest.offset - offset) <= RANGE_THRESHOLD:
            return nearest
    return None

def lookup_full(kb, db, offset, dtype):
    """All three tiers."""
    return kb.lookup(db, offset, dtype).symbol

CONFIGS = [
    ("仅 Tier-1",       lookup_tier1_only),
    ("Tier-1+2",        lookup_tier12),
    ("Tier-1+2+3 (完整)", lookup_full),
]

# ---------------------------------------------------------------- experiment
COVERAGE_LEVELS = [0.3, 0.5, 0.7, 1.0]

# Test queries: use ALL symbol addresses as ground truth
test_queries = [(s.db, s.offset, s.dtype, s.name) for s in all_symbols]

print(f"测试集: {len(test_queries)} 个地址查询（全部符号作为 ground truth）\n")
print(f"{'符号表完整度':>10}  {'配置':>18}  {'Precision@1':>12}  {'覆盖率':>8}")
print("-" * 58)

for coverage in COVERAGE_LEVELS:
    # Build a partial KB: keep `coverage` fraction of symbols, randomly
    n_keep = max(1, int(len(all_symbols) * coverage))
    kept = random.sample(all_symbols, n_keep)
    kb = KnowledgeBase()
    for s in kept:
        kb.add(s)

    for cfg_name, lookup_fn in CONFIGS:
        correct = 0
        answered = 0
        for db, offset, dtype, gt_name in test_queries:
            result = lookup_fn(kb, db, offset, dtype)
            if result is not None:
                answered += 1
                if result.name == gt_name:
                    correct += 1
        precision = correct / answered if answered else 0
        coverage_rate = answered / len(test_queries)
        print(f"{coverage*100:>9.0f}%  {cfg_name:>18}  {precision:>11.1%}  {coverage_rate:>7.1%}")
    print()
