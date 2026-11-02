"""Expected UART packets for fpga_viz/sim/tb_det_viz_uart.sv, from the detector's integer reference."""
import struct
import zlib
from pathlib import Path

import numpy as np

from fpga_det.client import MAGIC
from fpga_det.quantize import IntegerDetector
from . import ROOT, PROJECT
from .det_telemetry import PAYLOAD_BYTES, encode_block, reference_block

VECTORS = PROJECT / 'fpga_det' / 'export' / 'vectors'
OUT = ROOT / 'det_vectors'


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
    assert (version, kind, length) == (1, 0x10, 4 + 3 * 128 * 128)
    pixels = np.frombuffer(request[14 + 4:14 + length], dtype=np.uint8).reshape(128, 128, 3)
    q = (pixels.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
    out = IntegerDetector.load(PROJECT / 'fpga_det' / 'export').run(q, trace=True)
    expected_result = read_mem('image_response.mem')[14 + 2:14 + 10]
    assert struct.unpack('<4h', expected_result) == tuple(int(v) for v in out['class_logits'][0]), 'reference disagrees with image vector'
    payload = struct.pack('<I', image_frame) + encode_block(reference_block(out))
    assert len(payload) == PAYLOAD_BYTES
    early_id, late_id = 0x101, 0x202
    write_mem('telemetry_request_early.mem', packet(0x20, early_id))
    write_mem('telemetry_error_response.mem', packet(0xFF, early_id, bytes([4]) + bytes(15)))
    write_mem('telemetry_request.mem', packet(0x20, late_id))
    response = packet(0xA0, late_id, payload)
    write_mem('telemetry_response.mem', response)
    print('DET_VECTORS_WRITTEN image_frame', hex(image_frame), 'response_bytes', len(response), flush=True)


if __name__ == '__main__':
    main()
