"""CRC-framed USB UART client for the Basys 3 box detector. No laptop neural inference in hardware mode.

The board returns INT16 class/presence/size logits and the Q0.16 box centre; this client only
applies the final softmax/sigmoid to those 7 numbers and maps the box back to the frame.
"""
import argparse
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports
from PIL import Image, ImageOps

from detection.improve import letterbox
from . import ROOT

MAGIC = b"\xb3\x5d\xa7\xc1"
BAUD = 1000000
SIDE = 128
NAMES = ("Rubik's Cube", "Pen", "Book", "Computer Mouse")
RESULT = struct.Struct("<BB4hhHH2hI")


def encode(kind, frame_id, payload=b""):
    header = struct.pack("<BBII", 1, kind, frame_id, len(payload))
    return MAGIC + header + payload + struct.pack("<I", zlib.crc32(header + payload))


def model_id():
    return json.loads((ROOT / "export/hardware_identity.json").read_text())["model_id_hex"]


def preprocess(image):
    """EXIF-correct RGB, aspect-preserving bilinear letterbox to 128 with gray 128 padding (training geometry)."""
    return letterbox(ImageOps.exif_transpose(image).convert("RGB"), SIDE)[0]


def image_payload(letterboxed):
    return struct.pack("<HBB", SIDE, 1, 0) + np.asarray(letterboxed, dtype=np.uint8).tobytes()


def decode_result(data):
    cls, status, *rest = RESULT.unpack(data)
    logits, presence, cx16, cy16, size, cycles = rest[0:4], rest[4], rest[5], rest[6], rest[7:9], rest[9]
    z = np.array(logits) / 256.0; prob = np.exp(z - z.max()); prob /= prob.sum()
    p = 1 / (1 + np.exp(-presence / 256.0)); w, h = 1 / (1 + np.exp(-np.array(size) / 256.0))
    cx, cy = cx16 / 65536.0, cy16 / 65536.0
    return {"class_id": cls, "status": status, "name": NAMES[cls], "class_score": float(prob[cls]), "presence": float(p),
            "score": float(p * prob[cls]), "box_cxcywh_letterbox": [cx, cy, float(w), float(h)],
            "raw": {"class_logits_q8_8": list(logits), "presence_q8_8": presence, "cx_q16": cx16, "cy_q16": cy16, "size_q8_8": list(size)},
            "inference_cycles": cycles, "inference_ms": cycles / 100000, "inference_location": "Basys 3 FPGA"}


def find_port():
    ports = [p for p in list_ports.comports() if p.vid == 0x0403 and p.pid == 0x6010]
    if len(ports) != 1:
        raise RuntimeError(f"Expected one Basys 3 UART; found {[p.device for p in ports]}. Use --port explicitly.")
    return ports[0].device


class DetectorClient:
    def __init__(self, port=None, timeout=3.):
        self.port = port or find_port(); self.timeout = timeout; self.frame_id = 0
        self.serial = serial.Serial(self.port, BAUD, timeout=.1, write_timeout=5)
        self.serial.reset_input_buffer()
        try:
            self.identity = self.hello()
        except Exception:
            self.close(); raise

    def close(self):
        self.serial.close()

    def _read(self, n, deadline):
        result = bytearray()
        while len(result) < n:
            if time.perf_counter() > deadline:
                raise TimeoutError("FPGA response timeout; check that the DETECTOR bitstream is loaded, power and COM port")
            result.extend(self.serial.read(n - len(result)))
        return bytes(result)

    def receive(self):
        deadline = time.perf_counter() + self.timeout; window = bytearray()
        while bytes(window) != MAGIC:
            window.extend(self._read(1, deadline))
            if len(window) > 4:
                del window[0]
        header = self._read(10, deadline); version, kind, uid, length = struct.unpack("<BBII", header)
        if version != 1 or length > 256:
            raise ValueError("Invalid FPGA packet header")
        payload = self._read(length, deadline); crc = struct.unpack("<I", self._read(4, deadline))[0]
        if zlib.crc32(header + payload) != crc:
            raise ValueError("FPGA response CRC mismatch")
        return kind, uid, payload

    def request(self, kind, payload=b""):
        self.frame_id = (self.frame_id + 1) & 0xffffffff; uid = self.frame_id
        self.serial.write(encode(kind, uid, payload)); self.serial.flush()
        rkind, rid, result = self.receive()
        if rid != uid:
            raise ValueError("Stale/mismatched FPGA frame ID")
        if rkind == 0xff:
            raise RuntimeError(f"FPGA rejected packet (error code {result[0]})")
        return rkind, result

    def hello(self):
        kind, payload = self.request(1)
        if kind != 0x81 or len(payload) != 16:
            raise ValueError("Unexpected HELLO response")
        signature, side, classes, mode, mid = struct.unpack("<4sHBBQ", payload)
        if signature == b"B3C1":
            raise ValueError("The board is running the CLASSIFIER bitstream; load the detector (ProgramDetector.cmd)")
        if (signature, side, classes, mode) != (b"B3D1", SIDE, 4, 1):
            raise ValueError("Wrong FPGA design/protocol")
        if f"{mid:016x}" != model_id():
            raise ValueError("FPGA model does not match this integer export")
        return {"port": self.port, "baud": BAUD, "model_id_hex": model_id(), "input": [SIDE, SIDE, 3]}

    def detect_letterboxed(self, letterboxed):
        start = time.perf_counter(); kind, data = self.request(0x10, image_payload(letterboxed))
        elapsed = time.perf_counter() - start
        if kind != 0x91 or len(data) != RESULT.size:
            raise ValueError("Invalid detection response")
        result = decode_result(data)
        if result["status"] or result["class_id"] >= 4:
            raise ValueError("Invalid class/status returned by FPGA")
        result.update(roundtrip_ms=elapsed * 1000, frame_id=self.frame_id)
        return result

    def detect(self, image):
        return self.detect_letterboxed(preprocess(image))


def expected_raw(reference, letterboxed):
    q = (np.asarray(letterboxed, dtype=np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
    out = reference.run(q)
    return {"class_logits_q8_8": [int(v) for v in out["class_logits"][0]], "presence_q8_8": int(out["presence"][0]),
            "cx_q16": int(out["cx16"][0]), "cy_q16": int(out["cy16"][0]), "size_q8_8": [int(v) for v in out["size"][0]]}


def self_test(client, output):
    from .quantize import IntegerDetector
    ref = IntegerDetector.load(ROOT / "export"); rows = []
    cases = json.loads((ROOT / "export/vectors/cases.json").read_text())
    for case in cases:
        with Image.open(case["path"]) as image:
            lb = preprocess(image)
        expected = expected_raw(ref, lb); actual = client.detect_letterboxed(lb)
        assert actual["raw"] == expected, (case["id"], expected, actual["raw"])
        rows.append({"id": case["id"], **actual}); print("PASS hardware integer match", case["id"], json.dumps(actual["raw"]), flush=True)
    # Deliberate corruption must be rejected, then a clean request must recover.
    uid = 0x12345678; packet = bytearray(encode(1, uid)); packet[-1] ^= 1
    client.serial.write(packet); kind, rid, payload = client.receive()
    assert kind == 0xff and rid == uid and payload[0] == 1, "Bad CRC was not rejected"
    identity = client.hello()
    client.serial.write(encode(1, 123)[:7]); client.serial.flush(); time.sleep(.65)
    assert client.hello()["model_id_hex"] == identity["model_id_hex"], "Partial-packet recovery failed"
    report = {"hardware_identity": identity, "vectors": rows, "all_outputs_exact": True, "bad_crc_rejected": True,
              "partial_packet_recovery": True, "neural_inference_on_fpga": True,
              "note": "Bit-exact agreement with the integer reference on five webcam validation frames; not an accuracy benchmark."}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DETECTOR HARDWARE SELF TEST PASSED", flush=True)


def write_uart_vectors(cycles):
    """Golden UART packets for fpga_det/sim/tb_det_uart.sv, from the same encoder the client uses."""
    from fpga.quantize import write_mem
    from .quantize import IntegerDetector
    vectors = ROOT / "export" / "vectors"; mid = int(model_id(), 16)
    write_mem(vectors / "hello_request.mem", list(encode(1, 0x01020304)), 8)
    write_mem(vectors / "hello_response.mem", list(encode(0x81, 0x01020304, struct.pack("<4sHBBQ", b"B3D1", SIDE, 4, 1, mid))), 8)
    write_mem(vectors / "error_response.mem", list(encode(0xff, 0x01020304, bytes([1]) + bytes(15))), 8)
    write_mem(vectors / "format_request.mem", list(encode(0x10, 0x0a0b0c0d, struct.pack("<HBB", 96, 1, 0))), 8)
    write_mem(vectors / "format_response.mem", list(encode(0xff, 0x0a0b0c0d, bytes([3]) + bytes(15))), 8)
    q = np.array([int(line, 16) for line in (vectors / "input0.mem").read_text().split()], dtype=np.uint8).reshape(3, SIDE, SIDE)
    pixels = (q ^ 0x80).transpose(1, 2, 0)  # CHW pixel-128 -> HWC RGB bytes
    write_mem(vectors / "image_request.mem", list(encode(0x10, 0x55667788, struct.pack("<HBB", SIDE, 1, 0) + pixels.tobytes())), 8)
    out = IntegerDetector.load(ROOT / "export").run(q.view(np.int8))
    cls = int(np.argmax(out["class_logits"][0]))
    payload = RESULT.pack(cls, 0, *[int(v) for v in out["class_logits"][0]], int(out["presence"][0]), int(out["cx16"][0]),
                          int(out["cy16"][0]), *[int(v) for v in out["size"][0]], cycles)
    write_mem(vectors / "image_response.mem", list(encode(0x91, 0x55667788, payload)), 8)
    print("UART vectors written; image response", len(payload), "byte payload, cycles", cycles)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port"); p.add_argument("--image"); p.add_argument("--self-test", action="store_true")
    p.add_argument("--write-uart-vectors", type=int, metavar="CYCLES", help="Generate tb_det_uart golden packets (no board needed)")
    p.add_argument("--output", default=str(ROOT / "hardware_test.json"))
    a = p.parse_args()
    if a.write_uart_vectors is not None:
        write_uart_vectors(a.write_uart_vectors); return
    client = DetectorClient(a.port)
    try:
        print(json.dumps(client.identity), flush=True)
        if a.self_test:
            self_test(client, a.output)
        elif a.image:
            with Image.open(a.image) as image:
                print(json.dumps(client.detect(image), indent=2), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
