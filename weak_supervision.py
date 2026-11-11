"""Weak-supervision statistics for zero-symbol-table S7Comm inference."""
import sys, os, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

from s7comm_parser import parse_pcap


def collect_observations(pcap_paths):
    """Aggregate observed values per (db, offset, dtype) across pcap files."""
    series = defaultdict(list)
    for path in pcap_paths:
        for ev in parse_pcap(path):
            if not (ev.items and ev.values):
                continue
            for it, val in zip(ev.items, ev.values):
                if val is None:
                    continue
                series[(it.db_number, it.byte_offset,
                        it.transport_size)].append(float(val))
    return series


def _trend_features(vals):
    n = len(vals)
    if n < 2:
        return {"n": n, "step_ratio": 0.0, "unique_ratio": 1.0,
                "tv_ratio": 0.0, "autocorr1": 0.0}

    diffs = [vals[i+1] - vals[i] for i in range(n - 1)]
    abs_diffs = [abs(d) for d in diffs]
    vspan = (max(vals) - min(vals)) or 1e-9

    zero_moves = sum(1 for d in abs_diffs if d < 0.001 * vspan)
    big_jumps = sum(1 for d in abs_diffs if d > 0.20 * vspan)
    step_ratio = (zero_moves + big_jumps) / len(diffs)

    unique_ratio = len(set(round(v, 3) for v in vals)) / n

    tv_ratio = sum(abs_diffs) / vspan

    mean = sum(vals) / n
    denom = sum((v - mean) ** 2 for v in vals) or 1e-9
    num = sum((vals[i] - mean) * (vals[i+1] - mean) for i in range(n - 1))
    autocorr1 = num / denom

    return {"n": n, "step_ratio": round(step_ratio, 3),
            "unique_ratio": round(unique_ratio, 3),
            "tv_ratio": round(tv_ratio, 3),
            "autocorr1": round(autocorr1, 3)}


def compute_stats(vals):
    """Statistical + temporal feature vector for one address's observations.

    Pure function of the observed values — no symbol-table access.
    """
    n = len(vals)
    if n == 0:
        return None
    vmin, vmax = min(vals), max(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)

    # order of magnitude of the observed range — a coarse physical-quantity hint
    vspan = vmax - vmin
    magnitude = int(math.floor(math.log10(max(abs(vmax), 1e-9)))) if vmax > 0 else 0

    # fraction of integer-valued samples (status/mode codes tend to be integral)
    int_ratio = sum(1 for v in vals if abs(v - round(v)) < 1e-6) / n

    feats = {
        "n": n,
        "min": round(vmin, 3), "max": round(vmax, 3),
        "mean": round(mean, 3), "std": round(std, 3),
        "span": round(vspan, 3),
        "magnitude": magnitude,
        "cv": round(std / abs(mean), 3) if abs(mean) > 1e-9 else 0.0,  # coeff of variation
        "int_ratio": round(int_ratio, 3),
        "constant": vspan < 1e-6,
    }
    feats.update(_trend_features(vals))
    return feats


def classify_behavior(feats):
    """Heuristic label of temporal behavior (rule-based, for sanity-check only).

    The LLM will do the real semantic inference; this is just to verify the
    features are separable at all.
    """
    if feats is None or feats["n"] < 3:
        return "insufficient"
    if feats.get("constant"):
        return "constant"          # never changes: config / fixed setpoint
    if feats.get("step_ratio", 0) > 0.6 and feats.get("unique_ratio", 1) < 0.4:
        return "setpoint"          # piecewise-constant with steps
    if feats.get("tv_ratio", 99) < 3.0 and feats.get("autocorr1", 0) > 0.6:
        return "measured_smooth"   # smoothly drifting analog (high persistence)
    if feats.get("tv_ratio", 0) > 8.0:
        return "measured_noisy"    # jittery signal (flow/feed)
    return "dynamic"


# --------------------------------------------------------------------------
# Tier 4b: statistics -> semantics inference (zero symbol table).
# The ONLY inputs are wire-derived statistics. No unit, no name, no symbol
# table. Statistical-reasoning rules go in the prompt; zinc-hydrometallurgy
# domain knowledge is injected via RAG over the paper corpus.
# --------------------------------------------------------------------------

def describe_stats(feats) -> str:
    """Render a statistics feature vector as a natural-language observation."""
    behavior = classify_behavior(feats)
    beh_txt = {
        "setpoint":        "piecewise-constant, holding steady then stepping to "
                           "new levels (characteristic of an operator setpoint)",
        "measured_smooth": "smoothly drifting with high temporal persistence "
                           "(characteristic of a regulated measured value)",
        "measured_noisy":  "fluctuating rapidly around an operating point "
                           "(characteristic of a noisy flow-like measurement)",
        "constant":        "never changing (a fixed configuration constant)",
        "dynamic":         "varying continuously around an operating point",
        "insufficient":    "observed too few times to characterize",
    }.get(behavior, "varying over time")

    return (
        f"Observed over {feats['n']} samples. "
        f"Value range [{feats['min']}, {feats['max']}] "
        f"(order of magnitude 1e{feats['magnitude']}), "
        f"mean {feats['mean']}, coefficient of variation {feats['cv']}. "
        f"Temporal behavior: {beh_txt}. "
        f"Total-variation/range = {feats.get('tv_ratio', 0)}, "
        f"lag-1 autocorrelation = {feats.get('autocorr1', 0)}."
    )


def build_stat_query(feats) -> str:
    """Leakage-free retrieval query from statistics only (no unit, no name)."""
    behavior = classify_behavior(feats)
    kind = {
        "setpoint": "operator setpoint control target",
        "measured_smooth": "regulated process measurement",
        "measured_noisy": "fluctuating flow or feed measurement",
        "constant": "fixed configuration parameter",
    }.get(behavior, "process variable")
    # magnitude gives a coarse physical-quantity hint without revealing the unit
    mag_hint = {0: "small-magnitude", 1: "tens-range",
                2: "hundreds-range", 3: "thousands-range"}.get(feats["magnitude"],
                                                               "")
    return (f"zinc hydrometallurgy purification process {kind}, "
            f"{mag_hint} value around {feats['mean']}, "
            f"copper and cobalt removal instrumentation and control variables")


def build_stat_query_neutral(feats) -> str:
    """Stat-only retrieval query WITHOUT any process-stage prior.

    Unlike build_stat_query (which hard-codes the purification / copper-cobalt
    stage, valid for the real-data eval whose tags are all purification), this
    variant names only the wire-derived statistics — sign, magnitude, range and
    temporal behaviour. Used where the correct knowledge may live in a different
    plant stage (e.g. the multi-hop electrowinning scenario), so a stage prior
    would mis-steer retrieval.
    """
    behavior = classify_behavior(feats)
    kind = {
        "setpoint": "operator setpoint control target",
        "measured_smooth": "regulated process measurement",
        "measured_noisy": "fluctuating flow or feed measurement",
        "constant": "fixed configuration parameter",
    }.get(behavior, "process variable")
    mag_hint = {0: "small-magnitude", 1: "tens-range",
                2: "hundreds-range", 3: "thousands-range"}.get(feats["magnitude"],
                                                               "")
    span_hint = ""
    lo, hi = feats.get("min", 0), feats.get("max", 0)
    if lo is not None and hi is not None and 0 <= lo and hi <= 100:
        span_hint = "normalized 0-100 percent signal, "
    # Transport-layer routing key (NOT a symbol table): the observed data-block /
    # slave address groups signals by device, so an ambiguous 0-100 value can be
    # routed to its device's bridge note (e.g. DB3 -> Modbus skid) without any
    # variable name or unit. The physical quantity is still inferred downstream.
    device_hint = feats.get("_device_hint", "")
    device_hint = f"{device_hint}, " if device_hint else ""
    return (f"industrial PLC {device_hint}{kind}, {span_hint}{mag_hint} value "
            f"around {feats['mean']}, smooth non-negative analog signal")


def _reflect_retrieve(vkb, feats, query, model, base_url, max_rounds=2,
                      k=3, verbose=False):
    """Reflective iterative retrieval for the zero-symbol-table setting.

    retrieve -> LLM assesses whether the docs support inferring a physical
    quantity from these statistics -> if not, LLM rewrites the query -> retry.
    Returns (docs, rounds, final_query). Degrades to single-round when the LLM
    is unreachable (assessor defaults to 'sufficient').
    """
    import json as json_lib
    import urllib.request

    def _chat(prompt):
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2, "stream": False}
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json_lib.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json_lib.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception:
            return ""

    stat_desc = describe_stats(feats)
    docs, rounds = [], 0
    for rounds in range(1, max_rounds + 1):
        docs = vkb.retrieve(query, k=k)
        context = "\n\n".join(
            f"[{d['metadata']['source']}] {d['content'][:300]}" for d in docs
        ) or "(nothing retrieved)"
        verdict_raw = _chat(
            f"""You decide whether to run ONE more retrieval before inferring the
physical quantity of an unknown zinc-hydrometallurgy PLC variable.

Statistics: {stat_desc}

Retrieved knowledge:
{context}

STEP 1 — POINTER CHECK (do this first, it overrides everything else):
Scan every retrieved note for an explicit instruction to look elsewhere — phrases
like "you MUST consult", "refer to", "see reference table", "consult catalogue",
"listed in <named table/document>". If such a pointer names a SPECIFIC reference
(e.g. a table code like "ER-CH-07", or a named catalogue) AND the text of that
named reference is NOT already visible in the retrieved knowledge above, then the
retrieval is INSUFFICIENT. Set "missing" to the exact name/code of that pointed-to
reference (copy it verbatim) so it can be fetched next. This is the whole point of
a multi-hop lookup — do not skip it.

STEP 2 — otherwise judge INSUFFICIENT only if the text directly CONTRADICTS the
statistics (incompatible sign/magnitude) or is entirely off-topic.

STEP 3 — otherwise answer SUFFICIENT. Do not mark insufficient merely because the
text is generic or lacks the exact quantity name; partial background is fine.

Respond ONLY JSON: {{"sufficient": true/false, "missing": "exact reference name/code to fetch next, empty if sufficient"}}""")
        parsed_ok = False
        try:
            s, e = verdict_raw.find("{"), verdict_raw.rfind("}") + 1
            v = json_lib.loads(verdict_raw[s:e])
            sufficient = bool(v.get("sufficient"))
            missing = v.get("missing", "")
            parsed_ok = True
        except Exception:
            sufficient, missing = True, ""   # unreachable -> stop (single round)
        if verbose:
            print(f"    [round {rounds}] sufficient={sufficient} parsed_ok={parsed_ok} "
                  f"raw={verdict_raw[:120]!r} q={query[:60]!r}")
        if sufficient or rounds == max_rounds:
            break
        new_q = _chat(
            f"""Rewrite a vector-DB search query to retrieve better zinc-
hydrometallurgy knowledge for a variable with these statistics.
Previous query: {query}
What was missing: {missing}
Statistics: {stat_desc}
Return ONLY the new query text, one line.""").strip()
        query = (new_q.splitlines()[0].strip() if new_q
                 else (f"{query}; focus on {missing}" if missing else query))
    return docs, rounds, query


# Reference-code shapes seen in real industrial cross-references, most specific
# first so the longest structured code wins:
#   ER-CH-07                two+ letter groups + numeric  (catalogue/table codes)
#   GSD-DPV1-04             device-profile style codes
#   Table 7-3 / Tab. 7-3    "see Table X-Y" manual pointers (Modbus register maps)
#   LT-101 / FT-2201        ISA-5.1 instrument tags (P&ID -> datasheet hop)
_POINTER_PATTERNS = [
    r"\b[A-Z]{2,}(?:-[A-Z0-9]{1,})+-\d{1,4}\b",   # ER-CH-07, GSD-DPV1-04
    r"\b(?:Table|Tab\.?|Reference table)\s+\d{1,3}-\d{1,3}\b",  # Table 7-3
    r"\b[A-Z]{2}-\d{2,4}\b",                        # LT-101, FT-2201
]


def _extract_pointers(text):
    """Pull explicit reference codes out of retrieved note text.

    Returns a de-duplicated list of pointer codes in their ORIGINAL casing,
    longest-first so the most specific catalogue code is tried before a bare
    instrument tag. Original casing matters because the downstream exact-match
    fetch (ChromaDB $contains) is case-sensitive. De-duplication is done on an
    upper-cased key so "Table 7-3" and "TABLE 7-3" aren't both kept. Pure regex —
    no LLM, no model-judgment risk.
    """
    import re
    found, seen = [], set()
    for pat in _POINTER_PATTERNS:
        for m in re.finditer(pat, text):
            code = re.sub(r"\s+", " ", m.group(0)).strip()
            key = code.upper()
            if key not in seen:
                seen.add(key)
                found.append(code)
    # longest first: "ER-CH-07" beats a stray "CH-07" substring elsewhere
    found.sort(key=len, reverse=True)
    return found


def _pointer_retrieve(vkb, feats, query, model=None, base_url=None,
                      max_hops=2, k=3, verbose=False):
    """Pointer-following multi-hop retrieval (GraphRAG-style, rule-driven).

    Instead of the query-rewrite loop in _reflect_retrieve (which needs the LLM to
    (a) notice a pointer and (b) rewrite a query that still embeds-matches the
    hop-2 doc), this follows explicit reference codes deterministically:

      hop-1  vector-retrieve with the statistics query
      extract reference codes from the retrieved text (regex, no LLM)
      for each un-resolved code: exact-substring fetch it directly from the KB
                                 (col.get where_document $contains) and merge

    "Continue or not" is thus an objective rule (is there an unresolved pointer
    code?) rather than a subjective LLM verdict. Returns (docs, hops, pointers)
    where hops is how many retrieval rounds actually fired (1 = no pointer found,
    2 = one pointer followed). Signature mirrors _reflect_retrieve so the two are
    drop-in A/B comparable; model/base_url are accepted but unused (kept for a
    future LLM-fallback extractor).
    """
    docs = vkb.retrieve(query, k=k)
    seen = {d["metadata"]["source"] for d in docs}
    # Extract pointers in RETRIEVAL-RANK order: the top-ranked bridge doc is the
    # most relevant hop-1, so its reference code must be followed before codes
    # that merely co-occur in lower-ranked docs. Pooling all text and sorting by
    # length would let a longer but less-relevant code (from another chain's
    # bridge that also surfaced) hijack the hop. Within one doc, longest-first.
    pointers, _pseen = [], set()
    for d in docs:
        for code in _extract_pointers(d["content"]):
            if code.upper() not in _pseen:
                _pseen.add(code.upper())
                pointers.append(code)

    hops = 1
    followed = []
    for code in pointers:
        # Already fully resolved if the code's defining text is present in a doc
        # OTHER than the one that merely cited it. Cheap check: does any current
        # doc *define* it (code near start) vs just mention it? We simply fetch
        # and rely on the source-dedup below to drop docs we already have.
        try:
            res = vkb.col.get(where_document={"$contains": code})
        except Exception:
            continue
        added = False
        for src, content, meta in zip(res["ids"], res["documents"],
                                      res["metadatas"]):
            if meta["source"] in seen:
                continue
            docs.append({"content": content, "metadata": meta})
            seen.add(meta["source"])
            added = True
        if added:
            hops = 2
            followed.append(code)
        if hops == 2 and len(followed) >= (max_hops - 1):
            break

    if verbose:
        print(f"    [pointer] hops={hops} pointers={pointers} "
              f"followed={followed} docs={[d['metadata']['source'] for d in docs]}")
    return docs, hops, pointers


def _chat_ollama(prompt, model, base_url, temperature=0.2, timeout=60):
    """Single-turn Ollama chat completion; returns text ('' on failure)."""
    import json as json_lib
    import urllib.request
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "stream": False}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json_lib.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json_lib.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception:
        return ""


def _rerank_retrieve(vkb, feats, query, model=None, base_url=None,
                     k=3, pool=10, verbose=False):
    """Advanced-RAG listwise reranking (RankGPT-style).

    Standard "Advanced RAG" recipe: over-retrieve a candidate pool with the dense
    retriever, then re-rank it with a stronger model and keep the top-k. With no
    offline cross-encoder weights available, we use the LLM itself as a listwise
    reranker (RankGPT) — a STRONGER reranker than a small cross-encoder, so if it
    still cannot surface the decisive bridge document the point is only reinforced.

    The reranker sees ONLY the wire statistics (no unit/name), same leakage-free
    budget as every other mode. Signature mirrors _pointer_retrieve for drop-in
    A/B comparability. Returns (docs, rounds=1, query).
    """
    import re
    pool_docs = vkb.retrieve(query, k=pool)
    if not pool_docs:
        return [], 1, query
    stat_desc = describe_stats(feats)
    listing = "\n".join(
        f"[{i}] ({d['metadata'].get('source','?')}) {d['content'][:220]}"
        for i, d in enumerate(pool_docs))
    raw = _chat_ollama(
        f"""You are reranking retrieved knowledge passages by how useful each is
for inferring the physical quantity of an unknown industrial signal described
ONLY by these statistics (no name, no unit):

{stat_desc}

Candidate passages:
{listing}

Rank the passages from MOST to LEAST useful for identifying the signal's physical
quantity. Respond ONLY with a JSON list of the top {k} passage indices, e.g. [3,0,7].""",
        model, base_url)
    order = []
    try:
        nums = re.findall(r"\d+", raw[raw.find("["):raw.rfind("]") + 1])
        for n in nums:
            i = int(n)
            if 0 <= i < len(pool_docs) and i not in order:
                order.append(i)
    except Exception:
        order = []
    # fall back to dense order for anything the reranker omitted / on parse failure
    for i in range(len(pool_docs)):
        if i not in order:
            order.append(i)
    docs = [pool_docs[i] for i in order[:k]]
    if verbose:
        print(f"    [rerank] pool={len(pool_docs)} order={order[:k]} "
              f"docs={[d['metadata'].get('source') for d in docs]}")
    return docs, 1, query


def _agentic_retrieve(vkb, feats, query, model=None, base_url=None,
                      max_hops=2, k=3, verbose=False):
    """Agentic-RAG multi-hop: the LLM autonomously decides whether/what to re-search.

    This is the "let the agent decide" baseline for the pointer mechanism. Unlike
    _reflect_retrieve (which is steered by an explicit POINTER-CHECK instruction),
    the agent here gets NO hand-coded pointer heuristic: it freely judges from the
    retrieved text whether it has enough to name the physical quantity, and if not
    it formulates the next search query in natural language (ReAct-style
    observe->decide->act). The follow-up hop is a dense vector search on the
    agent-authored query — the decision is subjective LLM judgment, contrasting the
    objective reference-code rule of _pointer_retrieve. Returns (docs, hops, query).
    """
    import json as json_lib
    stat_desc = describe_stats(feats)
    docs = vkb.retrieve(query, k=k)
    seen = {d["metadata"]["source"] for d in docs}
    hops = 1
    for _ in range(max_hops - 1):
        context = "\n\n".join(
            f"[{d['metadata'].get('source','?')}] {d['content'][:300]}"
            for d in docs) or "(nothing retrieved)"
        raw = _chat_ollama(
            f"""You are an agent identifying the physical quantity of an unknown
industrial signal from its statistics plus retrieved knowledge. You may issue ONE
more search if it would help.

Statistics: {stat_desc}

Retrieved so far:
{context}

Decide autonomously: do you have enough to name the physical quantity, or should
you search once more? If you should search again, write the single most useful
next search query. Respond ONLY JSON:
{{"need_more": true/false, "next_query": "search text, empty if done"}}""",
            model, base_url)
        need_more, next_q = False, ""
        try:
            s, e = raw.find("{"), raw.rfind("}") + 1
            v = json_lib.loads(raw[s:e])
            need_more = bool(v.get("need_more"))
            next_q = (v.get("next_query") or "").strip()
        except Exception:
            need_more = False
        if verbose:
            print(f"    [agentic] hop={hops} need_more={need_more} "
                  f"next_q={next_q[:60]!r}")
        if not need_more or not next_q:
            break
        added = False
        for d in vkb.retrieve(next_q, k=k):
            if d["metadata"]["source"] not in seen:
                docs.append(d)
                seen.add(d["metadata"]["source"])
                added = True
        query = next_q
        if added:
            hops += 1
    return docs, hops, query


def infer_from_statistics(feats, vkb, model="qwen2.5:32b",
                          base_url="http://localhost:11434/v1",
                          use_rag=True, iterative=False, pointer=False,
                          entity=False, graph=None, db=None, offset=None,
                          rerank=False, agentic=False, graph_llm=False) -> dict:
    """Infer physical-quantity family from wire statistics + RAG domain knowledge.

    Honest zero-symbol-table setting: nothing here reads a symbol table. The
    statistical-reasoning rules live in the prompt; domain grounding comes from
    the retrieved paper chunks. Returns {"quantity", "is_setpoint",
    "range_estimate", "reasoning", "confidence"}.

    use_rag=False ablates the retrieval channel (statistics + rules only) to
    isolate whether the paper corpus helps or biases the inference.
    iterative=True enables reflective iterative retrieval (retrieve -> self-
    assess -> rewrite query -> retry) instead of a single retrieval round.
    pointer=True enables rule-driven pointer-following multi-hop retrieval
    (retrieve -> regex-extract reference codes -> exact-fetch the pointed-to
    doc); it needs no LLM judgment and takes precedence over iterative.
    entity=True enables entity-graph augmented retrieval (retrieve -> pull
    same-tank / downstream-process-step context via the entity graph); requires
    graph + db + offset. Precedence: pointer > entity > iterative.
    """
    import json as json_lib
    import urllib.request

    stat_desc = describe_stats(feats)
    query = build_stat_query(feats)
    rounds = 0
    if use_rag:
        if pointer:
            docs, rounds, _ptrs = _pointer_retrieve(vkb, feats, query,
                                                    model, base_url)
        elif graph_llm and graph is not None:
            from entity_graph import entity_augmented_retrieve_llm
            docs, rounds, _rel = entity_augmented_retrieve_llm(
                vkb, feats, query, graph=graph, db=db, offset=offset,
                model=model, base_url=base_url)
        elif entity and graph is not None:
            from entity_graph import entity_augmented_retrieve
            docs, rounds, _rel = entity_augmented_retrieve(
                vkb, feats, query, graph=graph, db=db, offset=offset)
        elif agentic:
            docs, rounds, query = _agentic_retrieve(vkb, feats, query,
                                                    model, base_url)
        elif rerank:
            docs, rounds, query = _rerank_retrieve(vkb, feats, query,
                                                   model, base_url)
        elif iterative:
            docs, rounds, query = _reflect_retrieve(vkb, feats, query,
                                                    model, base_url)
        else:
            docs = vkb.retrieve(query, k=3)
        context = "\n\n".join(
            f"[Source: {d['metadata']['source']}]\n{d['content']}" for d in docs
        )
        knowledge_block = (
            "You may use the retrieved zinc-hydrometallurgy domain knowledge "
            "below, but the statistical evidence takes priority.\n\n"
            f"Retrieved domain knowledge:\n{context}\n"
        )
    else:
        docs = []
        knowledge_block = ""

    prompt = f"""You are an industrial control systems reverse-engineering expert.
The symbol table is UNAVAILABLE: there is no variable name and no engineering
unit. You must infer meaning from the statistical behavior of the observed
values.

{knowledge_block}
Statistical observation of the unknown variable:
{stat_desc}

Reasoning rules (apply IN THIS ORDER — sign and magnitude are decisive):
- STEP 1 (behavior): Piecewise-constant with steps => a SETPOINT / control
  target, not a measurement. Report is_setpoint=true and stop.
- STEP 2 (SIGN — the strongest discriminator, check FIRST): if the range
  contains NEGATIVE values (min < 0), the quantity is an ELECTRODE POTENTIAL /
  VOLTAGE (mV), NOT a concentration/level/flow/feed. Concentration, level,
  flow, feed rate and pH are physically non-negative; only potential/voltage
  goes negative (e.g. cobalt-removal redox potential -700..0 mV). A negative
  range is near-certain evidence of "potential" — report it and do not fall
  back to concentration.
- STEP 3 (magnitude — a HARD constraint): reject any quantity inconsistent
  with the observed range:
    * range in the hundreds-to-thousands (1e2-1e3): large flow (m3/h), feed/
      dosing rate (kg/h, t/h), or high concentration (g/L).
    * range exactly bounded to 0-100 that drifts continuously: percentage
      LEVEL / valve opening / operating-condition (%), which is MORE likely
      than concentration for a smoothly-varying 0-100 signal — do not assume
      concentration by default.
    * small positive potential 0-60 (mV): copper-removal potential.
    * single-digit / tightly-bounded small values (0-14): pH, or low
      concentration — NEVER a large flow rate.
- STEP 4 (variability): high coefficient of variation (cv>0.3) with wide range
  => feed/dosing or flow (operator- or load-driven). Low cv (<0.1) with smooth
  drift => a tightly regulated measurement (concentration, pH, potential).
- STEP 5 (ANTI-BIAS — important): "concentration" is the most frequent term in
  the retrieved zinc-purification corpus, so it is tempting to default to it.
  Do NOT. Only choose concentration when the statistics positively support it
  (non-negative, g/L-consistent magnitude) AND no stronger signal (negative
  sign => potential; steady 0-100 % => level; high-cv hundreds => flow/feed)
  applies. When unsure between concentration and level/flow/potential, prefer
  the quantity whose sign and range fit best, not the corpus-frequent one.
- STEP 6: Do NOT invent a specific tank id or metal you cannot justify. Prefer
  the physical-quantity family (e.g. "ion_concentration", "flow_rate",
  "feed_rate", "electrode_potential", "tank_level") and lower confidence when
  the exact tag is not determinable.
- Your chosen quantity MUST be numerically consistent with the observed range
  AND sign; if it is not, pick a different quantity.

Respond ONLY with JSON (no other text):
{{"quantity": "physical_quantity_family_snake_case", "is_setpoint": true/false, "range_estimate": "min-max", "reasoning": "one sentence", "confidence": 0.0-1.0}}
"""

    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2, "stream": False}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json_lib.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json_lib.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}") + 1
        result = json_lib.loads(content[start:end])
        result["_query"] = query
        result["_sources"] = [d["metadata"]["source"] for d in docs]
        result["_rounds"] = rounds
        return result
    except Exception as e:
        return {"quantity": "unknown", "is_setpoint": False,
                "range_estimate": f"{feats['min']}-{feats['max']}",
                "reasoning": "", "confidence": 0.0, "error": str(e)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", nargs="+", help="pcap file(s) with time-series data")
    ap.add_argument("--min-obs", type=int, default=5,
                    help="min observations to compute stats")
    args = ap.parse_args()

    series = collect_observations(args.pcaps)
    usable = {k: v for k, v in series.items() if len(v) >= args.min_obs}
    print(f"Addresses observed: {len(series)}  with >={args.min_obs} obs: {len(usable)}\n")

    print(f"{'Address':<16}{'n':>4}{'range':>18}{'mag':>4}{'cv':>7}"
          f"{'step':>7}{'tv':>7}{'ac1':>7}  behavior")
    print("-" * 80)
    for (db, off, dt), vals in sorted(usable.items()):
        f = compute_stats(vals)
        rng = f"[{f['min']:.1f},{f['max']:.1f}]"
        print(f"DB{db}.{off:<11}{f['n']:>4}{rng:>18}{f['magnitude']:>4}"
              f"{f['cv']:>7.2f}{f.get('step_ratio',0):>7.2f}"
              f"{f.get('tv_ratio',0):>7.2f}{f.get('autocorr1',0):>7.2f}  {classify_behavior(f)}")
