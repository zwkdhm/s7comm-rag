"""Tiered symbol-table lookup for S7Comm addresses."""
import csv
import json
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np

RANGE_THRESHOLD = 8
TIER3_SIM_THRESHOLD = 0.15
USE_SENTENCE_TRANSFORMER = False
CALIB_PATH = os.environ.get(
    "CALIB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration.json"))


@dataclass
class Symbol:
    db: int
    offset: int
    dtype: str
    name: str
    unit: str = ""
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    description: str = ""


@dataclass
class LookupResult:
    symbol: Optional[Symbol]
    confidence: float
    tier: int           # 1-4, or 0 for miss
    calibrated: float = 0.0


# calibration loader

_CALIB = None


def _load_calibration() -> dict:
    global _CALIB
    if _CALIB is None:
        try:
            with open(CALIB_PATH, encoding="utf-8") as f:
                _CALIB = json.load(f)
        except Exception:
            _CALIB = {}   # uncalibrated: calibrated == raw confidence
    return _CALIB


def calibrate(tier: int, raw_conf: float) -> float:
    """Map a tier's raw confidence onto the shared P(correct) scale using the
    fitted calibration table. Falls back to the raw value when no table is
    present, so behaviour is unchanged until calibration.py has been run."""
    calib = _load_calibration()
    entry = calib.get(str(tier))
    if not entry:
        return raw_conf
    if entry["kind"] == "constant":
        return float(entry["value"])
    # isotonic: piecewise-linear interpolation over the fitted grid
    x, y = entry["x"], entry["y"]
    return float(np.interp(raw_conf, x, y))


class KnowledgeBase:
    def __init__(self):
        self._table: dict[tuple, Symbol] = {}   # (db, offset) → Symbol
        self._by_db: dict[int, list] = {}        # db → [Symbol, ...]
        self._embeddings = None                  # lazy, built on first Tier-3 call
        self._embed_list: list = []

    def load_csv(self, path: str):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.add(Symbol(
                    db=int(row["db"]),
                    offset=int(row["offset"]),
                    dtype=row.get("type", ""),
                    name=row["name"],
                    unit=row.get("unit", ""),
                    range_min=float(row["range_min"]) if row.get("range_min") else None,
                    range_max=float(row["range_max"]) if row.get("range_max") else None,
                    description=row.get("description", ""),
                ))

    def add(self, symbol: Symbol):
        self._table[(symbol.db, symbol.offset)] = symbol
        self._by_db.setdefault(symbol.db, [])
        lst = self._by_db[symbol.db]
        for i, s in enumerate(lst):
            if s.offset == symbol.offset:
                lst[i] = symbol
                break
        else:
            lst.append(symbol)
        self._embeddings = None   # invalidate cache

    def remove(self, db: int, offset: int):
        self._table.pop((db, offset), None)
        if db in self._by_db:
            self._by_db[db] = [s for s in self._by_db[db] if s.offset != offset]
        self._embeddings = None

    def __len__(self):
        return len(self._table)


    def lookup(self, db: int, offset: int, dtype: str = "",
               use_llm: bool = False, unit: str = "") -> LookupResult:
        # unit is not used — not recoverable from wire, keeping param for compat

        # Tier 1: exact
        sym = self._table.get((db, offset))
        if sym:
            return LookupResult(sym, 1.0, 1, calibrate(1, 1.0))

        # Tier 2: nearest in same DB within threshold
        if self._by_db.get(db):
            nearest = min(self._by_db[db], key=lambda s: abs(s.offset - offset))
            dist = abs(nearest.offset - offset)
            if dist <= RANGE_THRESHOLD:
                conf = round(1.0 - dist / (RANGE_THRESHOLD + 1), 2)
                return LookupResult(nearest, conf, 2, calibrate(2, conf))

        # Tier 3: semantic search within same DB
        if self._by_db.get(db):
            sym, conf = self._semantic_search(db, offset, dtype)
            if conf >= TIER3_SIM_THRESHOLD:
                return LookupResult(sym, conf, 3, calibrate(3, conf))

        # Tier 4: LLM inference
        if use_llm:
            sym = self._llm_inference(db, offset, dtype)
            if sym:
                return LookupResult(sym, 0.2, 4, calibrate(4, 0.2))

        return LookupResult(None, 0.0, 0, 0.0)

    def _llm_inference(self, db: int, offset: int, dtype: str) -> Optional[Symbol]:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from tier4_llm_inference import (VectorKnowledgeBase,
                                             llm_infer_semantic, build_query)
        except Exception:
            return None

        if not hasattr(self, "_vkb"):
            try:
                self._vkb = VectorKnowledgeBase()
            except Exception:
                self._vkb = None
        if self._vkb is None:
            return None

        # leakage-free query: wire fields only, no unit
        query = build_query(db, offset, dtype)
        docs = self._vkb.retrieve(query, k=3)
        # Real local LLM reads retrieved docs and infers a NAME only (falls back
        # to rule-based llm_infer_simple if the LLM endpoint is unreachable).
        r = llm_infer_semantic(db, offset, dtype, docs,
                               model=os.environ.get("TIER4_MODEL", "qwen2.5:32b"))
        if not r.get("name") or r["name"].startswith("unknown_"):
            from tier4_llm_inference_lite import llm_infer_simple
            doc_objs = [type("D", (), {"content": d["content"],
                                        "source": d["metadata"]["source"]})()
                        for d in docs]
            r = llm_infer_simple(db, offset, dtype, doc_objs)
        return Symbol(db=db, offset=offset, dtype=dtype,
                      name=r["name"], unit="",
                      description=r.get("description", ""))

    def _semantic_search(self, db: int, offset: int, dtype: str) -> tuple:
        self._ensure_embeddings()
        query = _text(Symbol(db=db, offset=offset, dtype=dtype,
                             name="", description=""))
        q_vec = _ngram_vec(query, self._vocab)
        sims = self._embeddings @ q_vec / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-9)
        best = int(np.argmax(sims))
        return self._embed_list[best], round(float(sims[best]), 3)

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        if USE_SENTENCE_TRANSFORMER:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            texts = [_text(s) for s in self._table.values()]
            self._embed_list = list(self._table.values())
            vecs = model.encode(texts, normalize_embeddings=True)
            self._embeddings = vecs
        else:
            self._embed_list = list(self._table.values())
            texts = [_text(s) for s in self._embed_list]
            self._vocab = _build_vocab(texts)
            self._embeddings = np.vstack([_ngram_vec(t, self._vocab) for t in texts])


# -------------------------------------------------------- n-gram helpers (Tier 3 fallback)

def _text(s: Symbol) -> str:
    return f"DB{s.db} {s.dtype} {s.name} {s.description} {s.unit}".lower()

def _build_vocab(texts: list, n=3) -> dict:
    grams = set()
    for t in texts:
        grams.update(t[i:i+n] for i in range(len(t) - n + 1))
    return {g: i for i, g in enumerate(sorted(grams))}

def _ngram_vec(text: str, vocab: dict, n=3) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for i in range(len(text) - n + 1):
        g = text[i:i+n]
        if g in vocab:
            vec[vocab[g]] += 1.0
    return vec
