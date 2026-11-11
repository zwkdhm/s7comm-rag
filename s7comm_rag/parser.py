"""S7Comm packet parser — handles Read/Write Var (function 0x04/0x05)."""
import struct
from dataclasses import dataclass
from typing import Optional

from scapy.all import rdpcap, TCP, Raw

AREA_MAP = {0x81: "I", 0x82: "Q", 0x83: "M", 0x84: "DB", 0x85: "DI", 0x1C: "C", 0x1D: "T"}
TYPE_MAP = {1: "BIT", 2: "BYTE", 3: "CHAR", 4: "WORD", 5: "INT", 6: "DWORD", 7: "DINT", 8: "REAL"}
TYPE_SIZE = {"BIT": 1, "BYTE": 1, "CHAR": 1, "WORD": 2, "INT": 2, "DWORD": 4, "DINT": 4, "REAL": 4}
FUNC_MAP = {0x04: "Read", 0x05: "Write"}


@dataclass
class S7Event:
    func: str           # "Read" | "Write"
    area: str           # "DB" | "M" | "I" | "Q" ...
    db: int
    offset: int         # byte offset
    bit: int            # bit offset (only meaningful for BIT type)
    dtype: str          # "REAL" | "WORD" | "INT" | ...
    count: int
    value: Optional[object] = None   # decoded value (Write requests only)
    timestamp: Optional[float] = None


def _decode_value(data: bytes, dtype: str) -> object:
    size = TYPE_SIZE.get(dtype, 0)
    if not data or len(data) < size:
        return data.hex() if data else None
    try:
        if dtype == "REAL":   return round(struct.unpack_from(">f", data)[0], 4)
        if dtype == "INT":    return struct.unpack_from(">h", data)[0]
        if dtype == "DINT":   return struct.unpack_from(">i", data)[0]
        if dtype == "DWORD":  return struct.unpack_from(">I", data)[0]
        if dtype in ("WORD", "BYTE", "CHAR"): return int.from_bytes(data[:size], "big")
        if dtype == "BIT":    return bool(data[0] & 0x01)
    except Exception:
        pass
    return data.hex()


def _parse_request_items(params: bytes):
    """Parse S7ANY request items; return (func_str, list_of_item_dicts)."""
    if len(params) < 2 or params[0] not in FUNC_MAP:
        return None, []
    func = FUNC_MAP[params[0]]
    item_count = params[1]
    items, pos = [], 2
    for _ in range(item_count):
        if pos + 12 > len(params):
            break
        spec_type, item_len = params[pos], params[pos + 1]
        if spec_type != 0x12 or item_len < 10:
            break
        # pos+2 is syntax_id (0x10 = S7ANY), skip it
        transport_size = params[pos + 3]
        count  = struct.unpack_from(">H", params, pos + 4)[0]
        db     = struct.unpack_from(">H", params, pos + 6)[0]
        area   = params[pos + 8]
        addr   = params[pos + 9: pos + 12]
        byte_addr = (addr[0] << 16 | addr[1] << 8 | addr[2]) >> 3
        bit_addr  = addr[2] & 0x07
        items.append({
            "func": func,
            "area": AREA_MAP.get(area, f"0x{area:02X}"),
            "db": db, "offset": byte_addr, "bit": bit_addr,
            "dtype": TYPE_MAP.get(transport_size, "UNKNOWN"),
            "count": count,
        })
        pos += 2 + item_len
    return func, items


def _parse_write_values(data: bytes, items: list) -> list:
    """Extract decoded values from Write Job data section."""
    values, pos = [], 0
    for item in items:
        if pos + 4 > len(data):
            values.append(None)
            continue
        transport_size = data[pos + 1]
        length_field   = struct.unpack_from(">H", data, pos + 2)[0]
        # length is in bits for transport_size 3 & 5, in bytes for 4
        byte_count = (length_field + 7) // 8 if transport_size in (3, 5) else length_field
        value_bytes = data[pos + 4: pos + 4 + byte_count]
        values.append(_decode_value(value_bytes, item["dtype"]))
        pos += 4 + byte_count + (byte_count % 2)   # word-align
    return values


def parse_pcap(pcap_path: str) -> list[S7Event]:
    """Return all S7Comm Read/Write Var events from a pcap file."""
    events = []
    for pkt in rdpcap(pcap_path):
        if TCP not in pkt or Raw not in pkt:
            continue
        raw = bytes(pkt[Raw])
        # TPKT starts with 0x03; skip TPKT(4) + COTP DT(3) = 7 bytes
        if len(raw) < 17 or raw[0] != 0x03:
            continue
        s7 = raw[7:]
        if len(s7) < 10 or s7[0] != 0x32:
            continue
        rosctr = s7[1]
        if rosctr not in (1, 3):    # Job or Ack-Data
            continue
        param_len = struct.unpack_from(">H", s7, 6)[0]
        data_len  = struct.unpack_from(">H", s7, 8)[0]
        hdr_size  = 10 if rosctr == 1 else 12
        params = s7[hdr_size: hdr_size + param_len]
        data   = s7[hdr_size + param_len: hdr_size + param_len + data_len]

        func, items = _parse_request_items(params)
        if not items:
            continue
        values = _parse_write_values(data, items) if func == "Write" else [None] * len(items)

        ts = float(pkt.time) if hasattr(pkt, "time") else None
        for item, val in zip(items, values):
            events.append(S7Event(
                func=item["func"], area=item["area"],
                db=item["db"], offset=item["offset"], bit=item["bit"],
                dtype=item["dtype"], count=item["count"],
                value=val, timestamp=ts,
            ))
    return events
