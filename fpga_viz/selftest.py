"""On-board check of the fpga_viz bitstream (no webcam needed).

1. HELLO identity, and telemetry refused before any inference.
2. The 3 reference photos: logits/score/class bit-exact with the integer reference (as before),
   AND the telemetry block bit-exact with the reference, AND tagged with the right frame ID.
3. 20 back-to-back frames: pairing and timing (what the recording view will see).
4. The original fpga.client still works unchanged against this bitstream.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fpga.client import FPGAClient, preprocess
from fpga.quantize import IntegerClassifier, classify_logits
from . import ROOT, PROJECT
from .client import VizClient
from .telemetry import reference_block, blocks_equal, expected_stamps


def main():
    reference = IntegerClassifier.load(PROJECT / 'fpga' / 'export')
    client = VizClient()
    report = {'port': client.port, 'identity': client.identity}
    try:
        try:
            client.telemetry(0)
            raise AssertionError('telemetry should be refused before any inference')
        except RuntimeError as error:
            assert 'error code 4' in str(error), error
        print('PASS telemetry refused before first inference', flush=True)
        rows = []
        for i in range(3):
            meta = json.loads((PROJECT / f'fpga/export/vectors/case{i}.json').read_text())
            with Image.open(meta['image']) as image:
                pixels = np.array(preprocess(image))
                q = (pixels.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
                logits, states, pooled = reference.run(q, trace=True)
                label, confidence = classify_logits(logits[0])
                result = client.classify(image)
            block = client.telemetry(result['frame_id'])
            assert list(result['logits_q8_8']) == list(logits[0]), (result, logits)
            assert result['class_id'] == label and result['confidence_u16'] == confidence
            assert blocks_equal(reference_block(states, pooled[0]), block), 'telemetry differs from reference'
            assert block['stamps'] == expected_stamps()
            rows.append({'case': i, 'class': result['name'], 'frame_id': result['frame_id'], 'telemetry_frame_id': block['frame_id']})
            print('PASS case', i, result['name'], 'logits+score+telemetry bit-exact, frame', block['frame_id'], flush=True)
        times = []
        rng = np.random.default_rng(1)
        for n in range(20):
            image = Image.fromarray(rng.integers(0, 256, (120, 120, 3), dtype=np.uint8))
            start = time.perf_counter()
            result = client.classify(image)
            mid = time.perf_counter()
            block = client.telemetry(result['frame_id'])
            end = time.perf_counter()
            pixels = np.array(preprocess(image))
            q = (pixels.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
            logits, states, pooled = reference.run(q, trace=True)
            assert list(result['logits_q8_8']) == list(logits[0]) and blocks_equal(reference_block(states, pooled[0]), block)
            times.append(((mid - start) * 1000, (end - mid) * 1000))
        t = np.array(times)
        report['round_trip_ms_median'] = float(np.median(t[:, 0]))
        report['telemetry_ms_median'] = float(np.median(t[:, 1]))
        print(f'PASS 20 random frames bit-exact; classify {np.median(t[:, 0]):.0f} ms, telemetry {np.median(t[:, 1]):.0f} ms (median)', flush=True)
        report['cases'] = rows
    finally:
        client.close()
    original = FPGAClient()
    try:
        meta = json.loads((PROJECT / 'fpga/export/vectors/case0.json').read_text())
        with Image.open(meta['image']) as image:
            result = original.classify(image)
        print('PASS original client still works:', result['name'], flush=True)
    finally:
        original.close()
    report['all_passed'] = True
    (ROOT / 'build' / 'hardware_selftest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VIZ HARDWARE SELF TEST PASSED', flush=True)


if __name__ == '__main__':
    main()
