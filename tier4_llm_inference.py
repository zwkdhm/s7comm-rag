"""Tier 4: LLM-based semantic inference via vector knowledge retrieval."""
import os
from dataclasses import dataclass
from typing import List, Optional

DEPLOYMENT_DOMAIN = "zinc hydrometallurgy purification"


def build_query(db: int, offset: int, dtype: str) -> str:
    return (f"{DEPLOYMENT_DOMAIN} process variable; "
            f"PLC data block DB{db} {dtype} at offset {offset}")


# 用 chromadb 做向量库（轻量，本地运行）
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    print("Warning: chromadb not installed. Run: pip install chromadb")


@dataclass
class KnowledgeDoc:
    """A knowledge document for vector retrieval."""
    content: str        # 文档内容
    source: str         # 来源（manual/paper/symbol_pattern）
    metadata: dict      # 元数据（协议类型、行业领域等）


class VectorKnowledgeBase:
    """Vector-based knowledge retrieval for Tier 4 LLM inference."""

    def __init__(self, persist_dir="data/vector_kb",
                 embed_model="/home/ind-bert/models/all-MiniLM-L6-v2"):
        if chromadb is None:
            raise RuntimeError("chromadb not installed")

        # Force offline mode — server has no internet
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.client = chromadb.PersistentClient(path=persist_dir)
        # Use local embedding model (download separately, no internet on server)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self.collection = self.client.get_or_create_collection(
            name="s7comm_knowledge",
            embedding_function=self.embed_fn
        )

    def add_documents(self, docs: List[KnowledgeDoc]):
        """批量添加知识文档到向量库。"""
        self.collection.add(
            documents=[d.content for d in docs],
            metadatas=[{"source": d.source, **d.metadata} for d in docs],
            ids=[f"doc_{i}" for i in range(len(docs))]
        )

    def retrieve(self, query: str, k: int = 3) -> List[dict]:
        """检索最相关的 k 条知识。"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]


# ----------------------------------------------------------------- 预置知识库构建

def build_s7comm_knowledge_base() -> List[KnowledgeDoc]:
    """构造 S7comm 领域知识库（示例，实际应从文档提取）。"""
    docs = []

    # 1. 协议规范知识
    docs.append(KnowledgeDoc(
        content=(
            "S7comm protocol uses Data Blocks (DB) to store PLC variables. "
            "DB1 typically contains global process variables like sensor readings. "
            "REAL type (4 bytes) represents floating-point measurements such as "
            "temperature (°C), pressure (bar), flow rate (m³/h). "
            "Offset 0-100 in DB1 usually stores primary process values."
        ),
        source="s7comm_spec",
        metadata={"type": "protocol", "confidence": "high"}
    ))

    # 2. 锅炉系统领域知识
    docs.append(KnowledgeDoc(
        content=(
            "Boiler control systems monitor: water temperature (inlet/outlet, °C), "
            "steam pressure (bar), fuel flow rate (m³/h), oxygen level (%), "
            "burner status (ON/OFF). Temperature setpoints are typically REAL type "
            "at DB1 offsets 4-8. Actual temperature readings at offsets 0-4."
        ),
        source="domain_boiler",
        metadata={"domain": "boiler", "confidence": "medium"}
    ))

    # 3. HVAC 系统知识
    docs.append(KnowledgeDoc(
        content=(
            "HVAC systems use PLCs to control: room temperature (°C), "
            "humidity (%), air flow (m³/s), valve positions (0-100%). "
            "Setpoints stored in DB2 offsets 0-20. Sensor readings in DB1. "
            "BOOL type for pump/fan status, REAL for analog values."
        ),
        source="domain_hvac",
        metadata={"domain": "hvac", "confidence": "medium"}
    ))

    # 3b. 传送带 / 物料输送系统知识
    docs.append(KnowledgeDoc(
        content=(
            "Conveyor belt systems in manufacturing PLCs typically use DB3 for "
            "material handling variables: DB3.DBD0 = conveyor_speed (m/s, REAL), "
            "DB3.DBD4 = conveyor_load (kg, REAL), DB3.DBW8 = conveyor_status "
            "(running/stopped/fault, WORD). Motor drives report speed and load "
            "as REAL values; status flags as WORD or BOOL."
        ),
        source="domain_conveyor",
        metadata={"domain": "conveyor", "confidence": "medium"}
    ))

    # 3c. 电机 / 驱动系统知识
    docs.append(KnowledgeDoc(
        content=(
            "Motor drive systems monitor: motor_speed (rpm or m/s, REAL), "
            "motor_current (A, REAL), motor_torque (Nm, REAL), motor_temperature "
            "(°C, REAL), running_status (BOOL). Speed and load feedback are REAL "
            "types; on/off and fault flags are BOOL or WORD."
        ),
        source="domain_motor",
        metadata={"domain": "motor", "confidence": "medium"}
    ))

    # 4. 通用命名模式（从历史符号表学习）
    docs.append(KnowledgeDoc(
        content=(
            "Common PLC variable naming patterns: "
            "DB1.DBD0 → actual_temperature (40% of projects), "
            "DB1.DBD4 → temperature_setpoint (32% of projects), "
            "DB1.DBD8 → pressure_sensor (25% of projects). "
            "DB2 often stores control outputs and setpoints. "
            "DB3 used for alarm and status flags."
        ),
        source="symbol_patterns",
        metadata={"type": "statistical", "confidence": "low"}
    ))

    # 5. 数据类型语义
    docs.append(KnowledgeDoc(
        content=(
            "S7comm data types and typical usage: "
            "REAL (4 bytes) → analog measurements (temp, pressure, flow). "
            "INT (2 bytes) → counters, scaled values (0-1000 → 0-100%). "
            "BOOL (1 bit) → binary status (pump ON/OFF, alarm active). "
            "BYTE (1 byte) → small integers, status codes."
        ),
        source="data_types",
        metadata={"type": "protocol", "confidence": "high"}
    ))

    return docs


# ----------------------------------------------------------------- LLM 推理接口

def llm_infer_semantic(db: int, offset: int, dtype: str,
                       retrieved_docs: List[dict],
                       model: str = "qwen2.5:32b",
                       base_url: str = "http://localhost:11434/v1",
                       stats: str = "") -> dict:
    import json as json_lib
    import urllib.request

    context = "\n\n".join([
        f"[Source: {d['metadata']['source']}]\n{d['content']}"
        for d in retrieved_docs
    ])

    stats_line = f"\n- Observed value statistics: {stats}" if stats else ""
    prompt = f"""You are an industrial control systems reverse-engineering expert.

Retrieved domain knowledge:
{context}

Task: infer the semantic meaning of an UNKNOWN PLC variable at:
- DB number: {db}
- Byte offset: {offset}
- Data type: {dtype}{stats_line}

Rules:
- Base your answer ONLY on the retrieved knowledge and the value statistics
  above. The engineering unit is NOT available (raw payloads carry no unit);
  do NOT assume or invent one.
- Infer the physical-quantity CATEGORY from the value range/magnitude (e.g.
  a small-magnitude smooth signal → ion_concentration or electrode_potential),
  giving a descriptive engineering term, NOT a placeholder like "db1_offset16".
- If the retrieved knowledge does not pin down the exact tank/stage, stay at
  the physical-quantity-category level (e.g. ion_concentration).

Respond ONLY with JSON (no other text):
{{"name": "descriptive_snake_case", "description": "brief"}}
"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json_lib.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json_lib.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        # extract JSON from possibly-wrapped response
        start, end = content.find("{"), content.rfind("}") + 1
        r = json_lib.loads(content[start:end])
        return {"name": r.get("name", ""), "description": r.get("description", "")}
    except Exception as e:
        return {"name": f"unknown_db{db}_offset{offset}", "description": "",
                "error": str(e)}


def _chat_llm(prompt: str, model: str, base_url: str, timeout: int = 60) -> str:
    import json as json_lib
    import urllib.request
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2, "stream": False}
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json_lib.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json_lib.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


def assess_retrieval(db: int, offset: int, dtype: str,
                     docs: List[dict], model: str = "qwen2.5:32b",
                     base_url: str = "http://localhost:11434/v1") -> dict:
    import json as json_lib
    context = "\n\n".join(
        f"[{d['metadata']['source']}] {d['content'][:300]}" for d in docs
    ) or "(no documents retrieved)"
    prompt = f"""You are auditing whether retrieved knowledge is RELEVANT enough
to infer the meaning of an unknown industrial PLC variable.

Unknown variable: DB{db} offset {offset} type {dtype}
Deployment domain: {DEPLOYMENT_DOMAIN}

Retrieved knowledge:
{context}

Judge strictly. The knowledge is "sufficient" ONLY if it names or describes a
physical quantity that plausibly matches this variable's domain. Generic
protocol text, or terms from an unrelated process stage, count as INSUFFICIENT.

Respond ONLY with JSON:
{{"sufficient": true/false, "reason": "one short sentence", "missing": "what keyword/topic to search for instead, empty if sufficient"}}"""
    content = _chat_llm(prompt, model, base_url)
    try:
        s, e = content.find("{"), content.rfind("}") + 1
        r = json_lib.loads(content[s:e])
        r["sufficient"] = bool(r.get("sufficient"))
        return r
    except Exception:
        return {"sufficient": True, "reason": "assessor unavailable", "missing": ""}


def rewrite_query(prev_query: str, missing: str, db: int, offset: int,
                  dtype: str, model: str = "qwen2.5:32b",
                  base_url: str = "http://localhost:11434/v1") -> str:
    prompt = f"""Rewrite a vector-DB search query to retrieve BETTER domain
knowledge for an unknown PLC variable.

Previous query: {prev_query}
Why it failed: {missing}
Variable facts: DB{db} offset {offset} type {dtype}
Domain: {DEPLOYMENT_DOMAIN}

Return ONLY the new search query text (one line, no quotes, no explanation)."""
    new_q = _chat_llm(prompt, model, base_url).strip()
    new_q = new_q.splitlines()[0].strip() if new_q else ""
    if not new_q:
        return f"{prev_query}; focus on {missing}" if missing else prev_query
    return new_q


def iterative_infer(vkb, db: int, offset: int, dtype: str,
                    k: int = 3, max_rounds: int = 2,
                    model: str = "qwen2.5:32b",
                    base_url: str = "http://localhost:11434/v1",
                    verbose: bool = False) -> dict:
    query = build_query(db, offset, dtype)
    docs, rounds = [], 0
    for rounds in range(1, max_rounds + 1):
        docs = vkb.retrieve(query, k=k)
        verdict = assess_retrieval(db, offset, dtype, docs, model, base_url)
        if verbose:
            print(f"  [round {rounds}] query={query!r}")
            print(f"            sufficient={verdict['sufficient']} "
                  f"reason={verdict.get('reason','')}")
        if verdict["sufficient"] or rounds == max_rounds:
            break
        query = rewrite_query(query, verdict.get("missing", ""),
                              db, offset, dtype, model, base_url)

    result = llm_infer_semantic(db, offset, dtype, docs, model=model,
                                base_url=base_url)
    result["rounds"] = rounds
    result["final_query"] = query
    return result


if __name__ == "__main__":
    vkb = VectorKnowledgeBase()
    docs = build_s7comm_knowledge_base()
    vkb.add_documents(docs)
    print(f"Added {len(docs)} documents to vector KB.\n")

    query = "DB1 offset 4 REAL type in industrial PLC, likely meaning"
    retrieved = vkb.retrieve(query, k=3)

    print("Retrieved:")
    for i, doc in enumerate(retrieved, 1):
        print(f"{i}. [{doc['metadata']['source']}] {doc['content'][:100]}...")
