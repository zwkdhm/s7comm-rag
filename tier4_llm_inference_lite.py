"""Tier 4: lightweight rule-based fallback inference (no external dependencies)."""
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KnowledgeDoc:
    content: str
    source: str
    keywords: List[str]


KNOWLEDGE_BASE = [
    KnowledgeDoc(
        content="S7comm DB1 typically stores global process variables. Offsets 0-100 contain primary measurements. REAL type (4 bytes) represents analog values like temperature (°C), pressure (bar), flow rate (m³/h).",
        source="s7comm_spec",
        keywords=["DB1", "REAL", "temperature", "pressure", "flow", "analog"]
    ),
    KnowledgeDoc(
        content="Boiler systems: DB1.DBD0 = actual_temperature (°C), DB1.DBD4 = temperature_setpoint (°C), DB1.DBD8 = pressure (bar), DB1.DBD12 = flow_rate (m³/h). REAL type for analog measurements.",
        source="domain_boiler",
        keywords=["boiler", "temperature", "setpoint", "pressure", "DB1", "REAL", "offset", "4", "8", "12"]
    ),
    KnowledgeDoc(
        content="HVAC systems: DB2 for setpoints, DB1 for sensor readings. Room temperature at DB1.DBD0 (°C), humidity at DB1.DBD4 (%), airflow at DB1.DBD8 (m³/s). BOOL for pump/fan status.",
        source="domain_hvac",
        keywords=["HVAC", "temperature", "humidity", "airflow", "DB1", "DB2", "BOOL", "pump", "fan"]
    ),
    KnowledgeDoc(
        content="Common patterns: DB1.DBD4 is temperature_setpoint in 32% of projects, flow_rate in 18%. DB1.DBD0 is actual_temperature in 45% of cases. DB2 stores control outputs.",
        source="statistical_patterns",
        keywords=["DB1", "DB2", "offset", "4", "0", "temperature", "setpoint", "pattern"]
    ),
    KnowledgeDoc(
        content="Data types: REAL (4 bytes) = floating-point measurements. INT (2 bytes) = counters, scaled values. BOOL (1 bit) = binary status. BYTE (1 byte) = status codes.",
        source="data_types",
        keywords=["REAL", "INT", "BOOL", "BYTE", "type", "measurement"]
    ),
]


def keyword_retrieve(query: str, k: int = 3) -> List[KnowledgeDoc]:
    query_lower = query.lower()
    scores = []
    for doc in KNOWLEDGE_BASE:
        score = sum(1 for kw in doc.keywords if kw.lower() in query_lower)
        scores.append((score, doc))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scores[:k] if score > 0]


def llm_infer_simple(db: int, offset: int, dtype: str, docs: List[KnowledgeDoc]) -> dict:
    """Rule-based fallback when the LLM endpoint is unavailable."""
    if db == 1 and offset == 4 and dtype == "REAL":
        return {"name": "temperature_setpoint", "description": "Temperature setpoint"}
    if db == 1 and offset == 0 and dtype == "REAL":
        return {"name": "actual_temperature", "description": "Actual temperature reading"}
    if db == 1 and offset == 8 and dtype == "REAL":
        return {"name": "pressure_sensor", "description": "Pressure sensor reading"}
    if db == 2 and dtype == "BOOL":
        return {"name": f"control_output_{offset}",
                "description": f"Control output at DB{db}.DBX{offset}"}
    if docs and "temperature" in docs[0].content.lower():
        return {"name": f"variable_db{db}_offset{offset}",
                "description": "Likely temperature-related variable"}
    return {"name": f"unknown_db{db}_offset{offset}",
            "description": f"Unknown variable at DB{db}.{dtype}{offset}"}


if __name__ == "__main__":
    query = "DB1 offset 4 REAL industrial PLC"
    docs = keyword_retrieve(query, k=3)
    result = llm_infer_simple(1, 4, "REAL", docs)
    print(result)
