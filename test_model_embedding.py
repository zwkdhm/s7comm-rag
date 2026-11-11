"""
Compare [CLS] embeddings from pretrained vs finetuned ET-BERT on S7Comm payloads.

Experiment:
  1. Parse synthetic pcap → S7 payloads (bigram tokenized)
  2. Encode with both models → [CLS] embeddings
  3. Compute intra-class vs inter-class cosine similarity
     (same semantic operation should cluster tighter after finetuning)
"""
import sys, os, binascii
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "fine-tuning"))

import torch
from uer.utils.constants import CLS_TOKEN
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from run_classifier import Classifier

# ------------------------------------------------------------------ config
MODELS = {
    "pretrained": os.path.join(BASE, "models/pretrained_model.bin"),
    "finetuned":  os.path.join(BASE, "models/finetuned_model_test.bin"),
}
VOCAB_PATH   = os.path.join(BASE, "models/encryptd_vocab.txt")
CONFIG_PATH  = os.path.join(BASE, "models/bert_base_config.json")
LABELS_NUM   = 10
SEQ_LENGTH   = 256
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ bigram tokenization (same as service)
def bigram_generation(hex_str: str, maxlen=128) -> str:
    chars = [hex_str[i:i+1] for i in range(len(hex_str))]
    tokens, count = [], 0
    for i in range(len(chars) - 1):
        if count >= maxlen:
            break
        tokens.append(chars[i] + chars[i+1])
        count += 1
    return " ".join(tokens)

def payload_to_text(raw_bytes: bytes) -> str:
    return bigram_generation(binascii.hexlify(raw_bytes).decode())

# ------------------------------------------------------------------ model loader
def load_etbert(model_path: str):
    import argparse
    args = argparse.Namespace()
    args.vocab_path        = VOCAB_PATH
    args.config_path       = CONFIG_PATH
    args.spm_model_path    = None
    args.labels_num        = LABELS_NUM
    args.seq_length        = SEQ_LENGTH
    args.encoder           = "transformer"
    args.embedding         = "word_pos_seg"
    args.mask              = "fully_visible"
    args.pooling           = "first"
    args.soft_targets      = False
    args.soft_alpha        = False
    args.factorized_embedding_parameterization = False
    args.parameter_sharing = False
    args.remove_transformer_bias   = False
    args.layernorm_positioning     = "post"
    args.feed_forward              = "dense"
    args.layernorm                 = "normal"
    args.relative_position_embedding       = False
    args.relative_attention_buckets_num    = 32
    args.share_relative_position_embedding = False
    args.remove_embedding_layernorm        = False
    args.remove_attention_scale            = False
    args.max_seq_length    = 512
    args.use_flash_attn    = False
    args.diff_noise_gate   = False
    args.diff_entropy_threshold  = 0.9
    args.diff_entropy_beta       = 8.0
    args.diff_local_band_strength= 0.0
    args.diff_local_band_width   = 3.0

    load_hyperparam(args)
    args.tokenizer = str2tokenizer["bert"](args)

    model = Classifier(args)
    model = load_model(model, model_path)
    model.eval()
    model.to(DEVICE)
    return model, args

# ------------------------------------------------------------------ embedding extraction
@torch.no_grad()
def get_embedding(model, args, text: str) -> np.ndarray:
    tokens = args.tokenizer.tokenize(text)
    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokens)
    if len(src) > args.seq_length:
        src = src[:args.seq_length]
    while len(src) < args.seq_length:
        src.append(0)
    seg = [1] * args.seq_length

    src_t = torch.tensor([src], dtype=torch.long).to(DEVICE)
    seg_t = torch.tensor([seg], dtype=torch.long).to(DEVICE)

    emb = model.embedding(src_t, seg_t)
    out = model.encoder(emb, seg_t)   # [1, seq_len, hidden]
    cls = out[:, 0, :].squeeze(0)     # [CLS] token
    vec = cls.cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-9)

# ------------------------------------------------------------------ pcap payload extraction
def collect_samples(pcap_path: str, scenario: str, max_per_pcap=8) -> list:
    """Return list of (label, full_packet_bytes) for S7Comm Read/Write packets.
    Uses full raw packet bytes for bigram tokenization (same as service pipeline).
    """
    from scapy.all import rdpcap, TCP, Raw
    from s7comm_parser import parse_pcap, _s7_offset
    import struct

    events = parse_pcap(pcap_path)
    pkts   = rdpcap(pcap_path)

    # Build map: packet index → event (match by order of S7Comm packets)
    s7_pkt_idx = []
    for i, pkt in enumerate(pkts):
        if pkt.haslayer(TCP) and pkt.haslayer(Raw):
            data = bytes(pkt[Raw])
            off  = _s7_offset(data)
            if off and len(data) > off + 10:
                func = data[off + 10]
                if func in (0x04, 0x05):  # Read or Write (skip Setup/handshake)
                    s7_pkt_idx.append(i)

    results = []
    for ev_i, (event, pkt_i) in enumerate(zip(events, s7_pkt_idx)):
        if ev_i >= max_per_pcap:
            break
        pkt = pkts[pkt_i]
        raw = bytes(pkt)           # full Ethernet frame for bigram
        item = event.items[0] if event.items else None
        addr = f"DB{item.db_number}[{item.byte_offset}]" if item else "?"
        label = f"{scenario}/{event.function}/{addr}"
        results.append((label, raw))
    return results


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    scenarios = {
        "boiler_startup":   "data/synthetic/boiler_startup.pcap",
        "pressure_control": "data/synthetic/pressure_control.pcap",
        "fault_scenario":   "data/synthetic/fault_scenario.pcap",
        "mixed_readwrite":  "data/synthetic/mixed_readwrite.pcap",
    }
    samples = []
    for scenario, path in scenarios.items():
        samples.extend(collect_samples(path, scenario, max_per_pcap=6))

    if not samples:
        print("No samples found. Run generate_test_data.py first.")
        sys.exit(1)

    texts = [payload_to_text(raw) for _, raw in samples]
    print(f"Samples: {len(samples)}\n")

    # Load and test both models
    results = {}
    for name, path in MODELS.items():
        print(f"Loading {name} from {os.path.basename(path)} ...")
        try:
            model, args = load_etbert(path)
            embeddings = np.array([get_embedding(model, args, t) for t in texts])
            results[name] = embeddings
            print(f"  Embedding shape: {embeddings.shape}")

            # Intra-scenario cosine similarity (same scenario = same semantic group)
            labels = [s.split("/")[0] for s, _ in samples]
            unique = list(dict.fromkeys(labels))
            intra, inter = [], []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = float(embeddings[i] @ embeddings[j])
                    (intra if labels[i] == labels[j] else inter).append(sim)

            intra_mean = np.mean(intra) if intra else 0
            inter_mean = np.mean(inter) if inter else 0
            print(f"  Intra-class similarity (same scenario): {intra_mean:.4f}")
            print(f"  Inter-class similarity (diff scenario): {inter_mean:.4f}")
            print(f"  Separation (intra - inter):             {intra_mean - inter_mean:.4f}\n")
        except Exception as e:
            print(f"  Failed: {e}\n")

    # Compare the two models
    if len(results) == 2:
        pre = results["pretrained"]
        fine = results["finetuned"]
        delta = np.mean(np.linalg.norm(fine - pre, axis=1))
        print(f"Mean L2 distance between pretrained and finetuned embeddings: {delta:.4f}")
        print("(Larger = finetuning shifted representations more)")
