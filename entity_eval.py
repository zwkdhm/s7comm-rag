"""Evaluate the ENTITY-relation multi-hop scenario: single vs entity-augmented.

Counterpart to multihop_eval.py (pointer scenario). Confusable 0-100 % signals
whose family (concentration vs potential) is fixed only by tank membership.
Single-round RAG cannot separate the pair; entity-graph augmentation pulls the
tank's role document via same-tank / process-step entity context and recovers it.

Ground truth (with `name`) is read from data/entity_symbols.csv; the EntityGraph
is built from the same CSV so tank membership parses correctly.
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
from entity_graph import EntityGraph

# neutral stat query (no purification prior baked in) for this scenario
weak_supervision.build_stat_query = build_stat_query_neutral

PCAP    = "data/synthetic_real/entity_poll.pcap"
CSV     = "data/entity_symbols.csv"
PERSIST = "data/vector_kb_entity"
COLL    = "entity_knowledge"
EMBED   = "/home/ind-bert/models/all-MiniLM-L6-v2"
MODEL   = os.environ.get("TIER4_MODEL", "qwen2.5:32b")


class EntityKB:
    """retrieve()-only KB over the isolated entity-scenario collection."""
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
    """Lexical BM25 retriever over the SAME entity-scenario documents (sparse
    baseline against the dense all-MiniLM-L6-v2 vectors). Okapi BM25, k1=1.5,
    b=0.75, over whitespace-lowercased tokens."""
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
    """mode: 'single' (one dense retrieval), 'entity' (entity-graph augmented),
    'norag' (no retrieval — statistics + rules only), or 'bm25' (single-round
    lexical BM25 retrieval instead of dense vectors)."""
    rows = {(int(r["db"]), int(r["offset"])): r for r in csv.DictReader(open(CSV))}
    series = collect_observations([PCAP])
    vkb = BM25KB() if mode == "bm25" else EntityKB()
    graph = EntityGraph().build(CSV)
    keys = sorted(k for k in series if (k[0], k[1]) in rows and len(series[k]) >= 5)

    entity = mode == "entity"
    graph_llm = mode == "graph-llm"
    rerank = mode == "rerank"
    use_rag = mode != "norag"
    print(f"Model: {MODEL}   Samples: {len(keys)}   RAG mode: {mode}\n")
    print(f"{'Address':<10}{'GT family':<15}{'pred family':<15}{'L3':>4}{'rounds':>7}")
    print("-" * 51)

    L3 = amb_hit = amb_tot = rounds_sum = total = 0
    # ambiguous = the confusable copper/cobalt pairs (DB1 conc, DB2 potential)
    AMBIGUOUS_DBS = {1, 2}
    for (db, off, dt) in keys:
        f = compute_stats(series[(db, off, dt)])
        gt_fam = rows[(db, off)]["gt_family"]
        r = infer_from_statistics(f, vkb, model=MODEL, use_rag=use_rag,
                                  entity=entity, graph=graph, db=db, offset=off,
                                  graph_llm=graph_llm, rerank=rerank)
        pred_fam = _norm_family(r.get("quantity", ""))
        rounds = r.get("_rounds", 0)
        hit = int(pred_fam == gt_fam)
        L3 += hit; rounds_sum += rounds; total += 1
        if db in AMBIGUOUS_DBS:
            amb_tot += 1; amb_hit += hit
        print(f"DB{db}.{off:<6}{gt_fam:<15}{pred_fam:<15}{hit:>4}{rounds:>7}")

    print("-" * 51)
    print(f"\nL3 Family hit-rate (n={total}): {100*L3//total}%")
    print(f"  ambiguous conc/potential subset (n={amb_tot}): {100*amb_hit//amb_tot}%")
    if entity or graph_llm:
        print(f"  avg retrieval rounds: {rounds_sum/total:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--entity", action="store_true",
                   help="entity-graph augmented retrieval (deterministic edges)")
    g.add_argument("--graph-llm", dest="graph_llm", action="store_true",
                   help="Graph-RAG baseline: LLM-extracted entity edges")
    g.add_argument("--rerank", action="store_true",
                   help="Advanced-RAG: over-retrieve + LLM listwise rerank (top-3)")
    g.add_argument("--norag", action="store_true",
                   help="no retrieval baseline (statistics + rules only)")
    g.add_argument("--bm25", action="store_true",
                   help="lexical BM25 single-round retrieval baseline")
    args = ap.parse_args()
    mode = ("entity" if args.entity else "graph-llm" if args.graph_llm
            else "rerank" if args.rerank else "norag" if args.norag
            else "bm25" if args.bm25 else "single")
    evaluate(mode=mode)
