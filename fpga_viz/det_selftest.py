"""On-board check of the detector + telemetry bitstream (no webcam needed)."""
import json
import time

import numpy as np
from PIL import Image

from fpga_det.client import DetectorClient, preprocess, expected_raw
from fpga_det.quantize import IntegerDetector
from . import ROOT, PROJECT
from .client import VizDetectorClient
from .det_telemetry import reference_block, blocks_equal, STAMPS


def main():
    reference = IntegerDetector.load(PROJECT / 'fpga_det' / 'export')
    client = VizDetectorClient()
    report = {'port': client.port, 'identity': client.identity}
    try:
        try:
            client.telemetry(0)
            raise AssertionError('telemetry should be refused before any inference')
        except RuntimeError as error:
            assert 'error code 4' in str(error), error
        print('PASS telemetry refused before first inference', flush=True)
        cases = json.loads((PROJECT / 'fpga_det/export/vectors/cases.json').read_text())
        for case in cases:
            with Image.open(case['path']) as image:
                lb = preprocess(image)
            q = (np.asarray(lb, dtype=np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
            out = reference.run(q, trace=True)
            result = client.detect_letterboxed(lb)
            block = client.telemetry(result['frame_id'])
            assert result['raw'] == expected_raw(reference, lb), case['id']
            assert blocks_equal(reference_block(out), block), f"telemetry differs for {case['id']}"
            assert block['stamps'] == STAMPS
            print('PASS', case['id'], result['name'], 'outputs + telemetry bit-exact', flush=True)
        rng = np.random.default_rng(2)
        times = []
        for _ in range(6):
            lb = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
            t0 = time.perf_counter(); result = client.detect_letterboxed(lb); t1 = time.perf_counter()
            block = client.telemetry(result['frame_id']); t2 = time.perf_counter()
            q = (np.asarray(lb, dtype=np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
            assert result['raw'] == expected_raw(reference, lb) and blocks_equal(reference_block(reference.run(q, trace=True)), block)
            times.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
        t = np.array(times)
        report['detect_ms_median'], report['telemetry_ms_median'] = float(np.median(t[:, 0])), float(np.median(t[:, 1]))
        print(f'PASS 6 random frames bit-exact; detect {np.median(t[:, 0]):.0f} ms, telemetry {np.median(t[:, 1]):.0f} ms (median)', flush=True)
    finally:
        client.close()
    original = DetectorClient()
    try:
        with Image.open(cases[0]['path']) as image:
            print('PASS original detector client still works:', original.detect(image)['name'], flush=True)
    finally:
        original.close()
    report['all_passed'] = True
    (ROOT / 'det_build' / 'hardware_selftest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DETECTOR VIZ HARDWARE SELF TEST PASSED', flush=True)


if __name__ == '__main__':
    main()
