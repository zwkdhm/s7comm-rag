# S7Comm Functional-Semantic Inference via RAG

A tiered retrieval-augmented inference system for translating raw S7Comm industrial protocol traffic into human-readable semantic descriptions, without requiring a complete symbol table.

## Overview

S7Comm (Siemens S7 protocol) packets carry only numeric DB addresses and byte offsets. When the symbol table is incomplete or unavailable, conventional tools cannot identify what a data point represents. This system resolves ambiguous addresses through a four-tier fallback chain, escalating from exact lookup to LLM-based statistical inference.

```
Tier 1 — Exact match       (DB + byte offset → direct symbol)
Tier 2 — Range match       (nearest entry within 8 bytes, same DB)
Tier 3 — Semantic search   (vector similarity over symbol descriptions)
Tier 4 — LLM inference     (time-series statistics + domain knowledge RAG)
```

Tiers 3 and 4 include two multi-hop retrieval strategies for cases where the decisive knowledge is not in a single document:

- **Pointer-following**: bridges an explicit cross-reference code to a role document
- **Entity-graph**: resolves ambiguous signals via device/tank membership context

## Repository Structure

```
s7comm_rag/          # core package (parser, knowledge base, translator)
knowledge_base.py    # four-tier KB with confidence scoring
calibration.py       # cross-tier confidence calibration (isotonic regression)
entity_graph.py      # entity-relation multi-hop retrieval
weak_supervision.py  # Tier-4 statistical inference (zero symbol table)
tier4_llm_inference.py      # LLM inference via Ollama (Qwen2.5)
tier4_llm_inference_lite.py # lightweight variant
demo.py              # end-to-end demo
data/
  symbols.csv              # minimal demo symbol table
  synthetic/               # synthetic S7Comm pcaps (boiler, pressure, fault)
  synthetic_real/          # zinc-hydrometallurgy scenario pcaps
  vector_kb/               # pre-built vector knowledge base (Chroma)
  vector_kb_multihop/      # pointer-following multi-hop KB
  vector_kb_entity/        # entity-graph multi-hop KB
```

## Requirements

```
Python >= 3.8
scapy >= 2.5.0
pandas >= 1.5.0
chromadb
sentence-transformers
ollama  # for Tier 4 LLM inference
```

Install dependencies:

```bash
pip install scapy pandas chromadb sentence-transformers
```

For Tier 4 inference, install [Ollama](https://ollama.ai) and pull the model:

```bash
ollama pull qwen2.5:7b
```

## Quick Start

```bash
# Run the end-to-end demo on a synthetic pcap
python demo.py
```

The demo parses a synthetic S7Comm capture, looks up each address through the four-tier chain, and prints the inferred semantic description alongside the confidence score.

## Experiments

The main evaluation scripts are:

| Script | Description |
|---|---|
| `eval_with_rag.py` | RAG paradigm comparison (Naïve / Advanced / Graph / Agentic) |
| `eval_degradation.py` | Symbol-table robustness under progressive coverage loss |
| `multihop_eval.py` | Pointer-following multi-hop disambiguation |
| `entity_eval.py` | Entity-graph multi-hop disambiguation |
| `eval_weak_supervision.py` | Zero-symbol-table inference (Tier 4 only) |
| `ablation_proper.py` | Ablation: remove innovations one by one |

## Configuration

Tier 4 LLM inference reads the Ollama endpoint from the environment:

```bash
export OLLAMA_HOST=http://localhost:11434
```

API keys for any external LLM service should be set via environment variable; see the top of `tier4_llm_inference.py` for the expected variable name.

## License

MIT License — see [LICENSE](LICENSE) for details.
