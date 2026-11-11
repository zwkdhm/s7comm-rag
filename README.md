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

Tiers 3 and 4 include two multi-hop retrieval strategies:

- **Pointer-following**: bridges an explicit cross-reference code to a role document
- **Entity-graph**: resolves ambiguous signals via device/tank membership context

## Repository Structure

```
s7comm_rag/              # core package
  knowledge.py           # four-tier KB with confidence scoring
  parser.py              # S7Comm packet parsing
  translator.py          # address-to-description translation
knowledge_base.py        # standalone KB entry point
s7comm_parser.py         # pcap parsing utilities
calibration.py           # cross-tier confidence calibration
entity_graph.py          # entity-relation multi-hop retrieval
weak_supervision.py      # zero-symbol-table statistical features
tier4_llm_inference.py   # LLM inference via Ollama
tier4_llm_inference_lite.py  # lightweight rule-based fallback
translator.py            # top-level translator
demo.py                  # end-to-end demo
calibration.json         # pre-fitted calibration table
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

For Tier 4 inference, install [Ollama](https://ollama.ai) and pull a model:

```bash
ollama pull qwen2.5:7b
```

## Quick Start

```bash
python demo.py
```

The demo parses a synthetic S7Comm capture, looks up each address through the four-tier chain, and prints the inferred semantic description alongside the confidence score.

## Configuration

Tier 4 LLM inference reads the Ollama endpoint from the environment:

```bash
export OLLAMA_HOST=http://localhost:11434
```

The calibration table path can be overridden:

```bash
export CALIB_PATH=/path/to/calibration.json
```

## License

MIT License — see [LICENSE](LICENSE) for details.
