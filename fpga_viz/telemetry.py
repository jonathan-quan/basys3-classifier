"""Layout of the FPGA telemetry block, its decoder, and the integer-reference expectation.

The block describes ONE inference and is read back over UART (request 0x20 -> 0xA0).
Payload = u32 frame ID of that inference + 2,836 bytes:
  0x000  9 x u32  cycle count when each conv layer finished
  0x024 48 x s8   global-average-pooled vector
  0x054 256 x u32 sum of every output channel (all 9 layers, 20 bits used)
  0x454 864 x u16 per-layer spatial heat map summed over channels
                  (layers 0-4: 12x12 bins, layers 5-8: 6x6 bins)
"""
import struct

import numpy as np

LAYER_CIN = (3, 8, 8, 16, 16, 32, 32, 48, 48)
LAYER_COUT = (8, 8, 16, 16, 32, 32, 48, 48, 48)
LAYER_KERNEL = (3, 3, 1, 3, 1, 3, 1, 3, 1)
LAYER_DEPTHWISE = (0, 1, 0, 1, 0, 1, 0, 1, 0)
LAYER_HW = (48, 24, 24, 12, 12, 6, 6, 6, 6)
LAYER_NAMES = ('conv 3x3 /2', 'dw 3x3 /2', 'pw 1x1', 'dw 3x3 /2', 'pw 1x1', 'dw 3x3 /2', 'pw 1x1', 'dw 3x3', 'pw 1x1')
CHANNEL_BASE = (0, 8, 16, 32, 48, 80, 112, 160, 208)
HEAT_GRID = (12, 12, 12, 12, 12, 6, 6, 6, 6)
HEAT_SHIFT = (2, 1, 1, 0, 0, 0, 0, 0, 0)
HEAT_BASE = (0, 144, 288, 432, 576, 720, 756, 792, 828)
STAMP_OFFSET, POOL_OFFSET, SUM_OFFSET, HEAT_OFFSET, BLOCK_BYTES = 0, 36, 84, 1108, 2836
PAYLOAD_BYTES = 4 + BLOCK_BYTES
TOTAL_CYCLES = 3674813  # whole inference incl. pooling, FC and softmax (fixed; data independent)
CLOCK_HZ = 100_000_000


def layer_cycle_counts():
    """Clock edges per conv layer, read straight off the core's FSM:
    LAYER + per output channel (3 parameter states + per pixel (BEGIN + 4 per tap + 3 requant))."""
    counts = []
    for cin, cout, k, dw, hw in zip(LAYER_CIN, LAYER_COUT, LAYER_KERNEL, LAYER_DEPTHWISE, LAYER_HW):
        taps = k * k * (1 if dw else cin)
        counts.append(1 + cout * (3 + hw * hw * (1 + 4 * taps + 3)))
    return counts


def expected_stamps():
    """inference_cycles sampled when layer_done is seen (2 cycles after the last write state)."""
    return [int(v) + 2 for v in np.cumsum(layer_cycle_counts())]


def reference_block(states, pooled):
    """What the telemetry block must contain for these integer-reference activations."""
    sums, heat = [], []
    for i, x in enumerate(states):
        x = np.asarray(x, dtype=np.int64)
        if x.ndim == 4:
            x = x[0]
        c, h, w = x.shape
        g, s = HEAT_GRID[i], 1 << HEAT_SHIFT[i]
        sums.append(x.sum(axis=(1, 2)))
        heat.append(x.reshape(c, g, s, g, s).sum(axis=(0, 2, 4)))
    return {'stamps': expected_stamps(), 'pooled': np.asarray(pooled, dtype=np.int64).reshape(-1),
            'channel_sums': sums, 'heat': heat}


def encode_block(block):
    out = bytearray(BLOCK_BYTES)
    struct.pack_into('<9I', out, STAMP_OFFSET, *block['stamps'])
    out[POOL_OFFSET:POOL_OFFSET + 48] = np.asarray(block['pooled'], dtype=np.int8).tobytes()
    struct.pack_into('<256I', out, SUM_OFFSET, *np.concatenate(block['channel_sums']).astype(np.uint32))
    struct.pack_into('<864H', out, HEAT_OFFSET, *np.concatenate([h.reshape(-1) for h in block['heat']]).astype(np.uint16))
    return bytes(out)


def decode_payload(payload):
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(f'Telemetry payload is {len(payload)} bytes, expected {PAYLOAD_BYTES}')
    frame_id = struct.unpack_from('<I', payload, 0)[0]
    data = payload[4:]
    stamps = list(struct.unpack_from('<9I', data, STAMP_OFFSET))
    pooled = np.frombuffer(data, dtype=np.int8, count=48, offset=POOL_OFFSET).astype(np.int64)
    flat_sums = np.array(struct.unpack_from('<256I', data, SUM_OFFSET), dtype=np.int64)
    flat_heat = np.array(struct.unpack_from('<864H', data, HEAT_OFFSET), dtype=np.int64)
    sums = [flat_sums[CHANNEL_BASE[i]:CHANNEL_BASE[i] + LAYER_COUT[i]] for i in range(9)]
    heat = [flat_heat[HEAT_BASE[i]:HEAT_BASE[i] + HEAT_GRID[i] ** 2].reshape(HEAT_GRID[i], HEAT_GRID[i]) for i in range(9)]
    return {'frame_id': frame_id, 'stamps': stamps, 'pooled': pooled, 'channel_sums': sums, 'heat': heat}


def blocks_equal(a, b):
    return (list(a['stamps']) == list(b['stamps']) and np.array_equal(a['pooled'], b['pooled'])
            and all(np.array_equal(x, y) for x, y in zip(a['channel_sums'], b['channel_sums']))
            and all(np.array_equal(x, y) for x, y in zip(a['heat'], b['heat'])))
