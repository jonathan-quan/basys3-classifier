"""Builds the expected UART packets for fpga_viz/sim/tb_viz_uart.sv from the integer reference."""
import struct
import zlib
from pathlib import Path

import numpy as np

from fpga.client import MAGIC
from fpga.quantize import IntegerClassifier
from . import ROOT, PROJECT
from .telemetry import PAYLOAD_BYTES, encode_block, reference_block, expected_stamps, TOTAL_CYCLES

VECTORS = PROJECT / 'fpga' / 'export' / 'vectors'
OUT = ROOT / 'vectors'


def packet(kind, frame_id, payload=b''):
    header = struct.pack('<BBII', 1, kind, frame_id, len(payload))
    return MAGIC + header + payload + struct.pack('<I', zlib.crc32(header + payload))


def read_mem(name):
    return bytes(int(v, 16) for v in (VECTORS / name).read_text().split())


def write_mem(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(''.join(f'{b:02x}\n' for b in data), encoding='ascii')


def main():
    request = read_mem('image_request.mem')
    version, kind, image_frame, length = struct.unpack('<BBII', request[4:14])
    assert (version, kind, length) == (1, 0x10, 27652)
    pixels = np.frombuffer(request[14 + 4:14 + length], dtype=np.uint8).reshape(96, 96, 3)
    q = (pixels.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
    reference = IntegerClassifier.load(PROJECT / 'fpga' / 'export')
    logits, states, pooled = reference.run(q, trace=True)
    expected_logits = struct.unpack('<4h', read_mem('image_response.mem')[14 + 4:14 + 12])
    assert list(logits[0]) == list(expected_logits), 'reference disagrees with the original image vector'
    stamps = expected_stamps()
    assert stamps[-1] < TOTAL_CYCLES
    block = encode_block(reference_block(states, pooled[0]))
    payload = struct.pack('<I', image_frame) + block
    assert len(payload) == PAYLOAD_BYTES
    early_id, late_id = 0x101, 0x202
    write_mem('telemetry_request_early.mem', packet(0x20, early_id))
    write_mem('telemetry_error_response.mem', packet(0xFF, early_id, bytes([4]) + bytes(15)))
    write_mem('telemetry_request.mem', packet(0x20, late_id))
    response = packet(0xA0, late_id, payload)
    write_mem('telemetry_response.mem', response)
    print('VECTORS_WRITTEN image_frame', image_frame, 'response_bytes', len(response), 'stamps', stamps)


if __name__ == '__main__':
    main()
