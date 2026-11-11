"""Evaluate the multi-hop disambiguation scenario: single vs iterative retrieval.

This is the counterpart to eval_weak_supervision.py, but on the deliberately
ambiguous multi-hop corpus (multihop_build_kb.py) and pcap (multihop_generate.py).
Ground-truth families are read explicitly from data/multihop_symbols.csv.

Expectation: single-round RAG cannot separate the ambiguous 0-100 potential/level
signals (the decisive rule is hop-2, unreachable in one round), whereas iterative
reflective retrieval bridges to hop-2 and recovers them.
"""
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(__file__))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.utils import embedding_functions
import weak_supervision
from weak_supervision import (collect_observations, compute_stats,
                              infer_from_statistics, build_stat_query_neutral)
from eval_weak_supervision import _norm_family

# The default build_stat_query hard-codes the purification/copper-cobalt stage,
# which mis-steers retrieval in this electrowinning multi-hop scenario. Use the
# neutral, stage-agnostic query builder instead (scoped to this eval only).
weak_supervision.build_stat_query = build_stat_query_neutral

PCAP    = "data/synthetic_real/multihop_poll.pcap"
CSV     = "data/multihop_symbols.csv"
PERSIST = "data/vector_kb_multihop"
COLL    = "multihop_knowledge"
EMBED   = "/home/ind-bert/models/all-MiniLM-L6-v2"
MODEL   = os.environ.get("TIER4_MODEL", "qwen2.5:32b")


class MultihopKB:
    """Minimal retrieve()-only KB over the isolated multi-hop collection."""
    def __init__(self):
        client = chromadb.PersistentClient(path=PERSIST)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED)
        self.col = client.get_collection(COLL, embedding_function=embed_fn)

    def retrieve(self, query, k=3):
        res = self.col.query(query_texts=[query], n_results=k)
        return [{"content": d, "metadata": m}
                for d, m in zip(res["documents"][0], res["metadatas"][0])]


class BM25KB:
    """Lexical BM25 retriever over the SAME multi-hop documents, as a
    sparse-retrieval baseline against the dense all-MiniLM-L6-v2 vectors.
    Pulls all docs+metadata once from the chroma collection, then scores
    with Okapi BM25 (k1=1.5, b=0.75) over whitespace-lowercased tokens."""
    def __init__(self):
        import math, re
        self._re = re.compile(r"[a-z0-9]+")
        client = chromadb.PersistentClient(path=PERSIST)
        got = client.get_collection(COLL).get(include=["documents", "metadatas"])
        self.docs = got["documents"]
        self.metas = got["metadatas"]
        self.toks = [self._tok(d) for d in self.docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(t) for t in self.toks) / max(self.N, 1)
        df = {}
        for t in self.toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for w, n in df.items()}

    def _tok(self, s):
        return self._re.findall(s.lower())

    def retrieve(self, query, k=3):
        q = self._tok(query)
        k1, b = 1.5, 0.75
        scored = []
        for i, toks in enumerate(self.toks):
            dl = len(toks)
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                s += self.idf.get(w, 0.0) * tf[w] * (k1 + 1) / (
                    tf[w] + k1 * (1 - b + b * dl / self.avgdl))
            scored.append((s, i))
        scored.sort(reverse=True)
        return [{"content": self.docs[i], "metadata": self.metas[i]}
                for _, i in scored[:k]]


def evaluate(mode: str = "single"):
    """mode: 'single' (one dense retrieval), 'iterative' (query-rewrite self-
    reflection), 'pointer' (rule-driven pointer-following multi-hop),
    'norag' (no retrieval — statistics + rules only), or 'bm25' (single-round
    lexical BM25 retrieval instead of dense vectors)."""
    rows = {(int(r["db"]), int(r["offset"])): r for r in csv.DictReader(open(CSV))}
    series = collect_observations([PCAP])
    vkb = BM25KB() if mode == "bm25" else MultihopKB()
    keys = sorted(k for k in series if (k[0], k[1]) in rows and len(series[k]) >= 5)

    # Transport-layer routing key: the data-block address groups signals by
    # device so an ambiguous 0-100 value routes to its device's bridge note.
    # This is wire-observable addressing, NOT a symbol table (no name/unit/semantic).
    DEVICE_HINT = {
        1: "electrowinning cellhouse data block DB1",
        3: "thickener underflow Modbus skid unit-id 3 data block DB3",
        4: "leaching train P&ID loop data block DB4",
    }
    # ambiguous 0-100% subsets that require the multi-hop reference to separate
    AMBIGUOUS_DBS = {1, 3, 4}

    iterative = mode == "iterative"
    pointer = mode == "pointer"
    rerank = mode == "rerank"
    agentic = mode == "agentic"
    use_rag = mode != "norag"
    print(f"Model: {MODEL}   Samples: {len(keys)}   RAG mode: {mode}\n")
    print(f"{'Address':<10}{'GT family':<15}{'pred family':<15}{'L3':>4}{'rounds':>7}")
    print("-" * 51)

    L3 = amb_hit = amb_tot = rounds_sum = total = 0
    for (db, off, dt) in keys:
        f = compute_stats(series[(db, off, dt)])
        if db in DEVICE_HINT:
            f["_device_hint"] = DEVICE_HINT[db]
        gt_fam = rows[(db, off)]["gt_family"]
        r = infer_from_statistics(f, vkb, model=MODEL, use_rag=use_rag,
                                  iterative=iterative, pointer=pointer,
                                  rerank=rerank, agentic=agentic)
        pred_fam = _norm_family(r.get("quantity", ""))
        rounds = r.get("_rounds", 0)
        hit = int(pred_fam == gt_fam)
        L3 += hit; rounds_sum += rounds; total += 1
        is_amb = db in AMBIGUOUS_DBS
        if is_amb:
            amb_tot += 1; amb_hit += hit
        print(f"DB{db}.{off:<6}{gt_fam:<15}{pred_fam:<15}{hit:>4}{rounds:>7}")

    print("-" * 51)
    print(f"\nL3 Family hit-rate (n={total}): {100*L3//total}%")
    print(f"  ambiguous 0-100 subset (n={amb_tot}): {100*amb_hit//amb_tot}%")
    if iterative or pointer or agentic:
        print(f"  avg retrieval rounds: {rounds_sum/total:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--iterative", action="store_true",
                   help="query-rewrite self-reflection multi-hop")
    g.add_argument("--pointer", action="store_true",
                   help="rule-driven pointer-following multi-hop")
    g.add_argument("--norag", action="store_true",
                   help="no retrieval baseline (statistics + rules only)")
    g.add_argument("--bm25", action="store_true",
                   help="lexical BM25 single-round retrieval baseline")
    g.add_argument("--rerank", action="store_true",
                   help="Advanced-RAG: over-retrieve + LLM listwise rerank (top-3)")
    g.add_argument("--agentic", action="store_true",
                   help="Agentic-RAG: LLM autonomously decides whether/what to re-search")
    args = ap.parse_args()
    mode = ("pointer" if args.pointer else "iterative" if args.iterative
            else "agentic" if args.agentic else "rerank" if args.rerank
            else "norag" if args.norag else "bm25" if args.bm25 else "single")
    evaluate(mode=mode)
