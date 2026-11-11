"""Test tier degradation after removing symbol table entries."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from s7comm_rag.knowledge import KnowledgeBase

TIER_LABEL = {1: "Tier-1 精确", 2: "Tier-2 范围", 3: "Tier-3 语义", 0: "未命中"}

def query(kb, db, offset, dtype="REAL"):
    r = kb.lookup(db, offset, dtype)
    sym = f"{r.symbol.name}({r.symbol.description})" if r.symbol else "None"
    return r.tier, r.confidence, sym

# ---------- 基准：完整符号表
kb_full = KnowledgeBase()
kb_full.load_csv("data/mock_symbols.csv")

# ---------- 删除后：移除 DB1.offset=4 和整个 DB2
kb_del = KnowledgeBase()
kb_del.load_csv("data/mock_symbols.csv")
kb_del.remove(1, 4)                          # 删除 BoilerTemp_SP
for off in [0, 4, 8, 12, 14]:               # 删除全部 DB2
    kb_del.remove(2, off)

print(f"完整符号表: {len(kb_full)} 条  删除后: {len(kb_del)} 条\n")

cases = [
    # (db, offset, dtype, 说明)
    (1,  4, "REAL", "DB1 offset=4  [已删除 BoilerTemp_SP]"),
    (1,  2, "REAL", "DB1 offset=2  [未删除，预期 Tier-2]"),
    (2,  0, "REAL", "DB2 offset=0  [DB2 全删，预期 Tier-3]"),
    (2,  8, "REAL", "DB2 offset=8  [DB2 全删，预期 Tier-3]"),
]

print(f"{'查询':35s}  {'完整KB':30s}  {'删除后':30s}")
print("-" * 100)
for db, off, dtype, desc in cases:
    t1, c1, s1 = query(kb_full, db, off, dtype)
    t2, c2, s2 = query(kb_del,  db, off, dtype)
    arrow = "⬇️ " if t2 > t1 else "  "
    print(f"{desc:35s}  {TIER_LABEL[t1]} {c1:.2f} {s1[:20]:20s}  "
          f"{arrow}{TIER_LABEL[t2]} {c2:.2f} {s2[:20]}")
