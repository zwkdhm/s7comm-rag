"""RAG translation pipeline: S7Event → natural language."""
import os
from dataclasses import dataclass
from s7comm_parser import S7Event
from knowledge_base import KnowledgeBase


@dataclass
class TranslationResult:
    event: S7Event
    translation: str
    confidence: float   # min confidence across all items
    tier: int           # max tier used (1=exact, 2=range, 3=semantic, 0=none)


class Translator:
    TIER_LABEL = {0: "无匹配", 1: "精确匹配", 2: "范围推断", 3: "语义检索"}

    def __init__(self, kb: KnowledgeBase,
                 llm_url: str = None, llm_key: str = "none", model: str = "gpt-4o"):
        self.kb = kb
        self._client = None
        if llm_url:
            from openai import OpenAI
            self._client = OpenAI(base_url=llm_url, api_key=llm_key)
        self.model = model

    # ---------------------------------------------------------------- public

    def translate(self, event: S7Event) -> TranslationResult:
        lookups = [
            self.kb.lookup(item.db_number, item.byte_offset, item.transport_size)
            for item in event.items
        ]
        conf = min((c for _, c, _ in lookups), default=0.0)
        tier = max((t for _, _, t in lookups), default=0)

        text = (self._llm(event, lookups) if self._client else None) \
               or self._template(event, lookups)
        return TranslationResult(event=event, translation=text, confidence=conf, tier=tier)

    def translate_all(self, events: list) -> list:
        return [self.translate(e) for e in events]

    # -------------------------------------------------------------- internal

    def _template(self, event: S7Event, lookups: list) -> str:
        parts = []
        vals = event.values or [None] * len(event.items)
        for item, val, (entry, conf, _) in zip(event.items, vals, lookups):
            addr = f"DB{item.db_number}.{item.transport_size}{item.byte_offset}"
            if entry and conf >= 0.5:
                name = entry.description or entry.tag_name
                unit = entry.unit or ""
                val_str = f" = {val}{unit}" if val is not None else ""
                parts.append(f"{name}({addr}){val_str}")
            else:
                val_str = f" = {val}" if val is not None else ""
                parts.append(f"未知参数({addr}){val_str}")
        joined = "；".join(parts)
        return f"{event.function}: {joined}"

    def _llm(self, event: S7Event, lookups: list) -> str | None:
        vals = event.values or [None] * len(event.items)
        items_ctx = []
        for item, val, (entry, conf, tier) in zip(event.items, vals, lookups):
            addr = f"DB{item.db_number}.{item.transport_size}{item.byte_offset}"
            ctx = {"address": addr}
            if val is not None:
                ctx["value"] = val
            if entry and conf >= 0.5:
                ctx.update({"tag": entry.tag_name, "description": entry.description,
                             "unit": entry.unit, "confidence": round(conf, 2)})
                if entry.min_val is not None:
                    ctx["range"] = f"{entry.min_val}~{entry.max_val}{entry.unit}"
            else:
                ctx["semantic"] = "UNKNOWN"
            items_ctx.append(ctx)

        prompt = (
            f"将以下S7Comm {event.function}报文翻译为一句简洁的中文操作描述。\n"
            f"参数: {items_ctx}\n"
            "要求: 包含操作类型、参数名、数值和单位；"
            "confidence<0.5或semantic=UNKNOWN的参数输出'未知参数(地址)'；"
            "不臆造未检索到的信息；直接输出描述句。"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120, temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] {e}")
            return None
