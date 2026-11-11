"""Parse S7Comm packets from pcap files into structured events."""
import struct
from dataclasses import dataclass, field
from typing import Optional

AREA_MAP = {0x81: "I", 0x82: "Q", 0x83: "M", 0x84: "DB", 0x1C: "C", 0x1D: "T"}
TRANSPORT_SIZE = {
    0x01: "BOOL", 0x02: "BYTE", 0x03: "CHAR",
    0x04: "WORD", 0x05: "INT",  0x06: "DWORD",
    0x07: "DINT", 0x08: "REAL",
}
FUNC_READ, FUNC_WRITE = 0x04, 0x05
ROSCTR_JOB, ROSCTR_ACK_DATA = 0x01, 0x03


@dataclass
class S7Item:
    area: str
    db_number: int
    byte_offset: int
    transport_size: str
    count: int


@dataclass
class S7Event:
    timestamp: float
    function: str
    items: list = field(default_factory=list)
    values: list = field(default_factory=list)


def _s7_offset(data: bytes) -> Optional[int]:
    if len(data) < 7 or data[0] != 0x03:
        return None
    cotp_len = data[4] + 1
    off = 4 + cotp_len
    if off < len(data) and data[off] == 0x32:
        return off
    return None


def _parse_address_item(data: bytes, off: int) -> tuple:
    if off + 12 > len(data):
        return None, off
    ts = TRANSPORT_SIZE.get(data[off + 3], "UNKNOWN")
    count = struct.unpack_from(">H", data, off + 4)[0]
    db    = struct.unpack_from(">H", data, off + 6)[0]
    area  = AREA_MAP.get(data[off + 8], f"0x{data[off+8]:02X}")
    addr  = (data[off+9] << 16) | (data[off+10] << 8) | data[off+11]
    return S7Item(area=area, db_number=db, byte_offset=addr >> 3,
                  transport_size=ts, count=count), off + 12


def _decode_raw(raw: bytes, ts: str):
    try:
        if ts in ("BOOL", "BYTE"):        return raw[0]
        if ts in ("WORD",):               return struct.unpack_from(">H", raw)[0]
        if ts == "INT":                   return struct.unpack_from(">h", raw)[0]
        if ts == "DWORD":                 return struct.unpack_from(">I", raw)[0]
        if ts == "DINT":                  return struct.unpack_from(">i", raw)[0]
        if ts == "REAL":                  return round(struct.unpack_from(">f", raw)[0], 4)
    except Exception:
        pass
    return raw.hex()


def _parse_data_item(data: bytes, off: int, ts: str) -> tuple:
    if off + 4 > len(data):
        return None, off
    length = struct.unpack_from(">H", data, off + 2)[0]
    byte_len = (length + 7) // 8
    raw = data[off + 4: off + 4 + byte_len]
    padded = byte_len + (byte_len % 2)
    return _decode_raw(raw, ts), off + 4 + padded


def parse_pcap(path: str) -> list:
    from scapy.all import rdpcap, TCP, Raw
    events = []
    try:
        pkts = rdpcap(path)
    except Exception as e:
        print(f"[parser] Cannot read {path}: {e}")
        return events

    for pkt in pkts:
        if not (pkt.haslayer(TCP) and pkt.haslayer(Raw)):
            continue
        data = bytes(pkt[Raw])
        s7off = _s7_offset(data)
        if s7off is None:
            continue

        s7 = data[s7off:]
        if len(s7) < 10:
            continue

        rosctr = s7[1]
        if rosctr not in (ROSCTR_JOB, ROSCTR_ACK_DATA):
            continue

        param_off = s7off + (12 if rosctr == ROSCTR_ACK_DATA else 10)
        if param_off >= len(data):
            continue

        func = data[param_off]
        if func not in (FUNC_READ, FUNC_WRITE):
            continue

        item_count = data[param_off + 1]
        off = param_off + 2

        items = []
        for _ in range(item_count):
            item, off = _parse_address_item(data, off)
            if item:
                items.append(item)

        values = []
        if func == FUNC_WRITE and rosctr == ROSCTR_JOB:
            for item in items:
                val, off = _parse_data_item(data, off, item.transport_size)
                values.append(val)

        if items:
            events.append(S7Event(
                timestamp=float(pkt.time),
                function="Write" if func == FUNC_WRITE else "Read",
                items=items,
                values=values,
            ))

    return events
