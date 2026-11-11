#!/usr/bin/env python3
"""Demo: translate S7Comm pcap events using symbol table RAG."""
import argparse, json, sys, os

sys.path.insert(0, os.path.dirname(__file__))

from s7comm_rag.parser import parse_pcap
from s7comm_rag.knowledge import KnowledgeBase
from s7comm_rag.translator import Translator


def main():
    ap = argparse.ArgumentParser(description="S7Comm RAG Translator")
    ap.add_argument("pcap", help="Path to .pcap file")
    ap.add_argument("--kb", default="data/mock_symbols.csv", help="Symbol table CSV")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    kb = KnowledgeBase()
    if os.path.exists(args.kb):
        kb.load_csv(args.kb)
        print(f"[知识库] 已加载 {len(kb)} 个符号", file=sys.stderr)
    else:
        print(f"[警告] 找不到知识库: {args.kb}", file=sys.stderr)

    events = parse_pcap(args.pcap)
    print(f"[解析] 共 {len(events)} 个 S7Comm 事件\n", file=sys.stderr)

    translator = Translator(kb)
    results = [translator.translate(e) for e in events]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            mark = "✓" if r["confidence"] == "high" else "?"
            print(f"[{mark}] {r['description']}")


if __name__ == "__main__":
    main()
