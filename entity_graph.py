"""Entity-relation multi-hop retrieval over the symbol table (in-memory graph)."""
import csv
import re
from collections import defaultdict


# controlled vocabularies grounded in zinc-hydrometallurgy symbol names
STAGES = ["leaching", "purification", "extract", "extraction", "extractive",
          "electrolysis"]
REMOVALS = ["copper", "cobalt", "cadmium", "nickel"]
MATERIALS = ["underflow", "supernatant", "seed", "powder", "waste", "acid"]
QUANTITIES = ["concentration", "flow", "level", "voltage", "potential",
              "temperature", "feed", "ph", "current", "density"]

# directed process flow: zinc hydrometallurgy main line + purification removal sequence
STAGE_FLOW = {
    "leaching":     ["purification"],
    "purification": ["extract", "extraction", "electrolysis"],
    "extract":      ["electrolysis"],
    "extraction":   ["electrolysis"],
}
REMOVAL_FLOW = {
    "copper":  ["cobalt"],
    "cobalt":  ["cadmium"],
    "cadmium": ["nickel"],
}


def _camel_to_snake(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", s).lower()


def parse_entity(name):
    """Parse a symbol-table name into entity fields (stage, removal, tank, quantity)."""
    if not name:
        return {}
    toks = set(re.split(r"[_\s]+", _camel_to_snake(name)))
    ent = {}
    for s in STAGES:
        if s in toks:
            ent["stage"] = "extract" if s in ("extraction", "extractive") else s
            break
    for r in REMOVALS:
        if r in toks:
            ent["removal"] = r
            break
    snake = _camel_to_snake(name)
    m = re.search(r"tank_([a-z]?\d+)", snake) or re.search(r"_(\d+)_", snake)
    if m:
        ent["tank"] = m.group(1)
    mats = [m for m in MATERIALS if m in toks]
    if mats:
        ent["material"] = mats
    quants = [q for q in QUANTITIES if q in toks]
    if quants:
        ent["quantity"] = quants
    return ent


class EntityGraph:
    """In-memory entity graph over the symbol table.

    Nodes are addresses (db, offset); edges connect addresses that share an
    entity relation: same removal-tank, same stage, or process-flow adjacency
    (stage flow + removal-sequence flow). Traversal from an address yields the
    related addresses whose documents/descriptions should augment retrieval.
    """

    def __init__(self):
        self.nodes = {}                 # (db,offset) -> {name, unit, desc, entity}
        self.adj = defaultdict(set)     # (db,offset) -> set of (db,offset)

    def build(self, symbols_csv):
        rows = list(csv.DictReader(open(symbols_csv, encoding="utf-8")))
        for r in rows:
            try:
                key = (int(r["db"]), int(r["offset"]))
            except (KeyError, ValueError):
                continue
            self.nodes[key] = {
                "name": r.get("name", ""),
                "unit": r.get("unit", ""),
                "desc": r.get("description", "") or r.get("cn_description", ""),
                "entity": parse_entity(r.get("name", "")),
            }
        self._build_edges()
        return self

    def _build_edges(self):
        keys = list(self.nodes)
        for i, a in enumerate(keys):
            ea = self.nodes[a]["entity"]
            for b in keys[i + 1:]:
                eb = self.nodes[b]["entity"]
                if self._related(ea, eb):
                    self.adj[a].add(b)
                    self.adj[b].add(a)

    @staticmethod
    def _related(ea, eb):
        """Graph edge = SAME TANK INSTANCE co-location (same removal + same tank
        id). This yields tight, meaningful neighbourhoods (the signals wired to
        one physical vessel). Process-flow across steps is intentionally NOT a
        dense edge here: at the category level copper<->cobalt / stage<->stage
        are many-to-many and would produce a near-complete graph (every copper
        node to every cobalt node). Cross-step flow is instead injected as a
        BOUNDED set of downstream context terms in `context_terms`.
        """
        if not ea or not eb:
            return False
        return bool(ea.get("removal") and ea.get("removal") == eb.get("removal")
                    and ea.get("tank") and ea.get("tank") == eb.get("tank"))

    def related_addresses(self, db, offset, hops=1):
        """BFS up to `hops` edges from (db,offset); return related addresses
        (excluding the seed)."""
        seed = (db, offset)
        if seed not in self.nodes:
            return []
        seen = {seed}
        frontier = {seed}
        for _ in range(hops):
            nxt = set()
            for node in frontier:
                for nb in self.adj[node]:
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            frontier = nxt
        seen.discard(seed)
        return sorted(seen)

    def related_entities(self, db, offset, hops=1):
        """Distinct entity descriptors of the related addresses (for building a
        context-augmenting sub-query)."""
        ents = []
        for k in self.related_addresses(db, offset, hops):
            e = self.nodes[k]["entity"]
            if e and e not in ents:
                ents.append(e)
        return ents

    def context_terms(self, db, offset, hops=1):
        """Entity terms for query augmentation: (a) the seed's OWN tank identity
        (stage/removal/material) — this steers retrieval to the seed's own
        per-tank role document; (b) same-tank neighbours' tank-identity ONLY
        (stage/removal/material); plus (c) a BOUNDED process-flow hint (the
        seed's downstream removal-step / stage). (a) is essential for
        disambiguation-by-tank: without it a copper-tank signal would only
        surface the downstream (cobalt) context.

        NO `quantity` field is ever added — neither the seed's own (it is the L3
        family being inferred) nor a same-tank neighbour's (which is the same
        family, e.g. a copper tank's other signals are also concentrations).
        Using either as a retrieval term would leak the answer family into the
        query. Only tank identity is used; the retrieved role doc supplies the
        quantity. (Verified inert on the real table, n=30: identical L3 with vs
        without the neighbour quantity — see REVISIONS.md problem 3.)
        """
        terms = []

        def _add(v):
            if v and v not in terms:
                terms.append(v)

        # (a) the seed's OWN tank identity (decisive for its tank role); no
        # quantity leak — see docstring.
        seed = self.nodes.get((db, offset))
        if seed:
            se = seed["entity"]
            _add(se.get("stage"))
            _add(se.get("removal"))
            for v in se.get("material", []):
                _add(v)

        # (b) same-tank neighbours — ONLY their tank-identity (stage/removal/
        # material), never their `quantity`. A same-tank neighbour's quantity is
        # the same L3 family being inferred (e.g. a copper tank's other signals
        # are also concentrations), so injecting it would leak the answer family
        # into the query. Verified inert: an A/B on the real table (n=30) gives
        # identical L3 (40%) with vs without it — role-doc fetch does the work,
        # so we drop it to make leakage-freedom hold by construction, not by the
        # target set happening not to trigger it.
        for k in self.related_addresses(db, offset, hops):
            e = self.nodes[k]["entity"]
            _add(e.get("stage"))
            _add(e.get("removal"))
            for v in e.get("material", []):
                _add(v)

        # (c) bounded downstream process-flow hint from the seed
        if seed:
            se = seed["entity"]
            for nxt in REMOVAL_FLOW.get(se.get("removal"), []):
                _add(nxt)
            for nxt in STAGE_FLOW.get(se.get("stage"), []):
                _add(nxt)
        return terms


def entity_augmented_retrieve(vkb, feats, query, graph=None, db=None,
                              offset=None, k=3, hops=1, verbose=False):
    """Entity-graph augmented retrieval.

    hop-1: vector-retrieve with the statistics query.
    augment (two ways, both keyed on the seed's tank identity from the graph):
      1. PRECISE role-doc fetch: exact-substring `<removal>-removal` against the
         KB (col.get $contains), analogous to pointer-following — reliable even
         when per-tank role docs are near-identical in embedding space.
      2. vector augment: a second query built from the entity context terms,
         as a soft fallback for docs without a literal tank phrase.
    Merge + dedup by source.

    Signature mirrors _pointer_retrieve for drop-in A/B comparability. Returns
    (docs, hops_used, related_terms). Degrades to plain single-round retrieval
    when no graph / no relations (hops_used=1).
    """
    docs = vkb.retrieve(query, k=k)
    seen = {d["metadata"]["source"] for d in docs}
    hops_used = 1
    related_terms = []

    if graph is not None and db is not None and offset is not None:
        related_terms = graph.context_terms(db, offset, hops=hops)

        # 1. precise role-doc fetch by the seed's own removal identity
        seed = graph.nodes.get((db, offset))
        removal = seed["entity"].get("removal") if seed else None
        if removal and getattr(vkb, "col", None) is not None:
            try:
                res = vkb.col.get(where_document={"$contains": f"{removal}-removal"})
                for content, meta in zip(res["documents"], res["metadatas"]):
                    if meta["source"] not in seen:
                        docs.append({"content": content, "metadata": meta})
                        seen.add(meta["source"])
                        hops_used = 2
            except Exception:
                pass

        # 2. soft vector augmentation from entity context terms
        if related_terms:
            aug_query = (f"{query}; related process context: "
                         f"{', '.join(related_terms)}")
            for d in vkb.retrieve(aug_query, k=k):
                if d["metadata"]["source"] not in seen:
                    docs.append(d)
                    seen.add(d["metadata"]["source"])
                    hops_used = 2

    if verbose:
        print(f"    [entity] hops={hops_used} related={related_terms} "
              f"docs={[d['metadata']['source'] for d in docs]}")
    return docs, hops_used, related_terms


def entity_augmented_retrieve_llm(vkb, feats, query, graph=None, db=None,
                                  offset=None, model="qwen2.5:32b",
                                  base_url="http://localhost:11434/v1",
                                  k=3, verbose=False):
    """Graph-RAG baseline whose entity relations are extracted by the LLM.

    This is the "generic Graph RAG" contrast to entity_augmented_retrieve: instead
    of taking the tank identity from the deterministically-parsed symbol table, it
    asks the LLM to read hop-1 documents and extract the entity/relation (which
    process tank/stage this signal belongs to). That LLM-extracted entity is then
    used as the $contains role-doc fetch key. Everything downstream is identical,
    so the ONLY changed variable is the graph-edge SOURCE: symbol-table parse vs.
    LLM text extraction. Noisy/mis-extracted entities are expected to degrade the
    role-doc hit-rate, isolating that the gain comes from deterministic edges, not
    from "using a graph" per se.

    Signature mirrors entity_augmented_retrieve. Returns (docs, hops_used, [entity]).
    """
    import json as json_lib
    from weak_supervision import _chat_ollama, describe_stats

    docs = vkb.retrieve(query, k=k)
    seen = {d["metadata"]["source"] for d in docs}
    hops_used = 1

    context = "\n\n".join(
        f"[{d['metadata'].get('source','?')}] {d['content'][:300]}"
        for d in docs) or "(nothing retrieved)"
    # LLM extracts the process-tank entity this signal belongs to (the graph edge).
    # It sees only wire statistics + retrieved text — no symbol-table name.
    raw = _chat_ollama(
        f"""From the retrieved zinc-hydrometallurgy knowledge and the signal
statistics, extract which purification removal stage/tank this unknown signal most
likely belongs to (e.g. copper, cobalt, cadmium). This is a graph-linking step:
name the single most likely removal entity so we can fetch that tank's role
document.

Statistics: {describe_stats(feats)}

Retrieved knowledge:
{context}

Respond ONLY JSON: {{"removal": "copper|cobalt|cadmium|nickel|unknown"}}""",
        model, base_url)
    removal = ""
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        removal = (json_lib.loads(raw[s:e]).get("removal") or "").strip().lower()
    except Exception:
        removal = ""

    if removal and removal != "unknown" and getattr(vkb, "col", None) is not None:
        try:
            res = vkb.col.get(where_document={"$contains": f"{removal}-removal"})
            for content, meta in zip(res["documents"], res["metadatas"]):
                if meta["source"] not in seen:
                    docs.append({"content": content, "metadata": meta})
                    seen.add(meta["source"])
                    hops_used = 2
        except Exception:
            pass

    if verbose:
        print(f"    [graph-llm] hops={hops_used} extracted_removal={removal!r} "
              f"docs={[d['metadata']['source'] for d in docs]}")
    return docs, hops_used, [removal]


if __name__ == "__main__":
    import os
    g = EntityGraph().build(os.path.join(os.path.dirname(__file__),
                                         "data/real_symbols.csv"))
    n_edges = sum(len(v) for v in g.adj.values()) // 2
    print(f"nodes={len(g.nodes)}  edges={n_edges}")
    # spot-check a few addresses' related set
    for (db, off) in list(g.nodes)[:5]:
        ent = g.nodes[(db, off)]["entity"]
        rel = g.related_addresses(db, off)
        terms = g.context_terms(db, off)
        print(f"\nDB{db}.{off}  entity={ent}")
        print(f"  #related={len(rel)}  context_terms={terms[:10]}")
