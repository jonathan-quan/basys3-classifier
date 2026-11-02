"""Telemetry block of the box detector (fpga_viz/det_rtl/detector_telemetry.sv).

Payload = u32 frame ID of the inference + 6,796 bytes:
  0     11 x u32   cycle count when each backbone layer finished
  44    96 x s8    global-average-pooled vector
  140  704 x u32   sum of every output channel (20 bits used)
  2956 1664 x u16  per-layer spatial heat map summed over channels (layers 0-4 16x16, 5-10 8x8)
  6284 256 x u16   softmax weight of each cell of the head's 16x16 box heat map (attention)
"""
import struct

import numpy as np

LAYER_COUT = (16, 16, 32, 32, 64, 64, 96, 96, 96, 96, 96)
LAYER_HW = (64, 32, 32, 16, 16, 8, 8, 8, 8, 8, 8)
CHANNEL_BASE = tuple(int(v) for v in np.concatenate([[0], np.cumsum(LAYER_COUT)[:-1]]))
HEAT_GRID = (16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8)
HEAT_SHIFT = (2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
HEAT_BASE = tuple(int(v) for v in np.concatenate([[0], np.cumsum([g * g for g in HEAT_GRID])[:-1]]))
STAMPS = [7405619, 8077412, 10338501, 10674470, 12853735, 13021864, 14625737, 14877930, 17268235, 17520428, 19910733]
STAMP_OFFSET, POOL_OFFSET, SUM_OFFSET, HEAT_OFFSET, ATT_OFFSET, BLOCK_BYTES = 0, 44, 140, 2956, 6284, 6796
PAYLOAD_BYTES = 4 + BLOCK_BYTES
assert CHANNEL_BASE[-1] + LAYER_COUT[-1] == 704 and HEAT_BASE[-1] + 64 == 1664


def reference_block(trace):
    """Expected block from IntegerDetector.run(q, trace=True) for ONE image."""
    sums, heat = [], []
    for i, x in enumerate(trace['states']):
        x = np.asarray(x, dtype=np.int64)
        if x.ndim == 4:
            x = x[0]
        c, h, w = x.shape
        g, s = HEAT_GRID[i], 1 << HEAT_SHIFT[i]
        sums.append(x.sum(axis=(1, 2)))
        heat.append(x.reshape(c, g, s, g, s).sum(axis=(0, 2, 4)))
    attention = np.asarray(trace['exp'], dtype=np.int64).reshape(-1, 256)[0].reshape(16, 16)
    return {'stamps': list(STAMPS), 'pooled': np.asarray(trace['pooled'], dtype=np.int64).reshape(-1, 96)[0],
            'channel_sums': sums, 'heat': heat, 'attention': attention}


def encode_block(block):
    out = bytearray(BLOCK_BYTES)
    struct.pack_into('<11I', out, STAMP_OFFSET, *block['stamps'])
    out[POOL_OFFSET:POOL_OFFSET + 96] = np.asarray(block['pooled'], dtype=np.int8).tobytes()
    struct.pack_into('<704I', out, SUM_OFFSET, *np.concatenate(block['channel_sums']).astype(np.uint32))
    struct.pack_into('<1664H', out, HEAT_OFFSET, *np.concatenate([h.reshape(-1) for h in block['heat']]).astype(np.uint16))
    struct.pack_into('<256H', out, ATT_OFFSET, *np.asarray(block['attention']).reshape(-1).astype(np.uint16))
    return bytes(out)


def decode_payload(payload):
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(f'Telemetry payload is {len(payload)} bytes, expected {PAYLOAD_BYTES}')
    frame_id = struct.unpack_from('<I', payload, 0)[0]
    data = payload[4:]
    stamps = list(struct.unpack_from('<11I', data, STAMP_OFFSET))
    pooled = np.frombuffer(data, dtype=np.int8, count=96, offset=POOL_OFFSET).astype(np.int64)
    flat_sums = np.array(struct.unpack_from('<704I', data, SUM_OFFSET), dtype=np.int64)
    flat_heat = np.array(struct.unpack_from('<1664H', data, HEAT_OFFSET), dtype=np.int64)
    attention = np.array(struct.unpack_from('<256H', data, ATT_OFFSET), dtype=np.int64).reshape(16, 16)
    sums = [flat_sums[CHANNEL_BASE[i]:CHANNEL_BASE[i] + LAYER_COUT[i]] for i in range(11)]
    heat = [flat_heat[HEAT_BASE[i]:HEAT_BASE[i] + HEAT_GRID[i] ** 2].reshape(HEAT_GRID[i], HEAT_GRID[i]) for i in range(11)]
    return {'frame_id': frame_id, 'stamps': stamps, 'pooled': pooled, 'channel_sums': sums, 'heat': heat, 'attention': attention}


def blocks_equal(a, b):
    return (list(a['stamps']) == list(b['stamps']) and np.array_equal(a['pooled'], b['pooled'])
            and all(np.array_equal(x, y) for x, y in zip(a['channel_sums'], b['channel_sums']))
            and all(np.array_equal(x, y) for x, y in zip(a['heat'], b['heat']))
            and np.array_equal(a['attention'], b['attention']))
