"""Tiered knowledge base for S7Comm semantic lookup.

Tier 1 — Exact:    (db, byte_offset) → direct hash lookup
Tier 2 — Range:    same DB, nearest entry within 8 bytes
Tier 3 — Semantic: embedding similarity over all entries
"""
import csv
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SymbolEntry:
    db_number: int
    byte_offset: int
    data_type: str
    tag_name: str
    description: str
    unit: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None


class KnowledgeBase:
    def __init__(self):
        self._exact: dict[tuple, SymbolEntry] = {}   # (db, offset) → entry
        self._by_db: dict[int, list] = {}             # db → [entries]
        self._embed_entries: list = []
        self._embeddings = None
        self._model = None

    # ------------------------------------------------------------------ load

    def load_csv(self, path: str):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                e = SymbolEntry(
                    db_number=int(row["db_number"]),
                    byte_offset=int(row["byte_offset"]),
                    data_type=row["data_type"],
                    tag_name=row["tag_name"],
                    description=row["description"],
                    unit=row.get("unit", ""),
                    min_val=float(row["min_val"]) if row.get("min_val") else None,
                    max_val=float(row["max_val"]) if row.get("max_val") else None,
                )
                self._exact[(e.db_number, e.byte_offset)] = e
                self._by_db.setdefault(e.db_number, []).append(e)

    def update_entry(self, entry: SymbolEntry, old_description: str = None):
        """Add or update a symbol entry. Detects semantic drift if old_description given."""
        key = (entry.db_number, entry.byte_offset)
        if old_description and key in self._exact:
            drift = self._semantic_distance(old_description, entry.description)
            if drift < 0.5:  # low similarity = high drift
                return {"status": "drift_warning",
                        "message": f"语义漂移告警: {key} 描述变化过大，需人工确认",
                        "old": old_description, "new": entry.description,
                        "similarity": round(drift, 3)}
        self._exact[key] = entry
        self._by_db.setdefault(entry.db_number, [])
        # replace or append in db list
        lst = self._by_db[entry.db_number]
        for i, e in enumerate(lst):
            if e.byte_offset == entry.byte_offset:
                lst[i] = entry
                break
        else:
            lst.append(entry)
        self._embeddings = None  # invalidate cache
        return {"status": "ok"}

    # --------------------------------------------------------- tiered lookup

    def lookup(self, db: int, offset: int, data_type: str = "") -> tuple:
        """Return (SymbolEntry | None, confidence 0-1, tier 1-3 | 0)."""

        # Tier 1: exact
        key = (db, offset)
        if key in self._exact:
            return self._exact[key], 1.0, 1

        # Tier 2: nearest within same DB (≤8 bytes)
        if db in self._by_db:
            nearest = min(self._by_db[db], key=lambda e: abs(e.byte_offset - offset))
            dist = abs(nearest.byte_offset - offset)
            if dist <= 8:
                return nearest, max(0.3, 1.0 - dist * 0.09), 2

        # Tier 3: semantic embedding search
        if self._exact:
            try:
                self._ensure_embeddings()
                query = f"DB{db} offset{offset} {data_type}"
                q = self._model.encode([query], normalize_embeddings=True)[0]
                sims = self._embeddings @ q
                best = int(np.argmax(sims))
                return self._embed_entries[best], float(sims[best]), 3
            except Exception:
                pass

        return None, 0.0, 0

    # --------------------------------------------------- semantic drift check

    def _semantic_distance(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text strings (0=dissimilar, 1=identical)."""
        self._ensure_embeddings()
        vecs = self._model.encode([text_a, text_b], normalize_embeddings=True)
        return float(vecs[0] @ vecs[1])

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self._embed_entries = list(self._exact.values())
        texts = [f"{e.tag_name} {e.description} {e.unit}" for e in self._embed_entries]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
