"""RAG translation pipeline: parser event + knowledge lookup → natural language."""
from typing import Callable, Optional

from .parser import S7Event
from .knowledge import KnowledgeBase, Symbol


def _addr_str(event: S7Event) -> str:
    if event.area == "DB":
        prefix = {"REAL": "D", "DWORD": "D", "DINT": "D",
                  "WORD": "W", "INT": "W", "BYTE": "B", "BIT": "X"}.get(event.dtype, "")
        return f"DB{event.db}.DB{prefix}{event.offset}"
    return f"{event.area}{event.offset}"


def _template(event: S7Event, sym: Optional[Symbol]) -> str:
    addr = _addr_str(event)
    tag  = sym.name if sym else "UNKNOWN"
    unit = f" {sym.unit}" if sym and sym.unit else ""
    hint = f" ({sym.description})" if sym and sym.description else ""
    if not sym:
        hint = f" (not in symbol table: {addr})"

    if event.func == "Write":
        val = f" = {event.value}{unit}" if event.value is not None else ""
        return f"Write {addr} [{tag}]{val}{hint}"
    elif event.func == "Read":
        return f"Read {addr} [{tag}]{unit}{hint}"
    return f"{event.func} {addr} [{tag}]{hint}"


class Translator:
    """
    Core RAG pipeline.

    llm_fn: optional hook for LLM-enhanced generation.
            signature: (event: S7Event, sym: Symbol | None) -> str
    """
    def __init__(self, kb: KnowledgeBase, llm_fn: Optional[Callable] = None):
        self.kb = kb
        self.llm_fn = llm_fn

    def translate(self, event: S7Event) -> dict:
        sym = self.kb.lookup(event.db, event.offset)
        if self.llm_fn:
            description = self.llm_fn(event, sym)
        else:
            description = _template(event, sym)

        return {
            "func":        event.func,
            "address":     _addr_str(event),
            "dtype":       event.dtype,
            "value":       event.value,
            "tag":         sym.name if sym else None,
            "unit":        sym.unit if sym else None,
            "description": description,
            "confidence":  "high" if sym else "low",
        }
