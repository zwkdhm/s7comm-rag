"""Verify three-tier retrieval with controlled test cases."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from s7comm_rag.knowledge import KnowledgeBase

TIER_LABEL = {1: "Tier-1 精确", 2: "Tier-2 范围", 3: "Tier-3 语义", 0: "未命中"}
PASS, FAIL = "✅", "❌"

kb = KnowledgeBase()
kb.load_csv("data/mock_symbols.csv")
print(f"[KB] 加载 {len(kb)} 个符号\n")

cases = [
    # (db, offset, dtype, expected_tier, desc)
    (1,  4,  "REAL", 1, "DB1 offset=4  → 精确命中 BoilerTemp_SP"),
    (1,  6,  "REAL", 2, "DB1 offset=6  → 范围推断，最近邻 offset=4 或 offset=8，距离≤2"),
    (1,  3,  "REAL", 2, "DB1 offset=3  → 范围推断，最近邻 offset=4，距离=1"),
    (99, 0,  "REAL", 3, "DB99 offset=0 → 无同DB条目，语义检索兜底"),
    (5,  100,"WORD", 3, "DB5 offset=100→ 无同DB条目，语义检索兜底"),
]

all_pass = True
for db, off, dtype, expected_tier, desc in cases:
    r = kb.lookup(db, off, dtype)
    ok = r.tier == expected_tier
    all_pass &= ok
    sym_str = f"{r.symbol.name}({r.symbol.description})" if r.symbol else "None"
    print(f"{PASS if ok else FAIL} {desc}")
    print(f"   → {TIER_LABEL[r.tier]}  conf={r.confidence:.2f}  match={sym_str}\n")

print("=" * 55)
print(f"{'全部通过 ✅' if all_pass else '存在失败项 ❌'}")
