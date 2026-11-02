"""FPGA client with telemetry readback.

Image packets and results use the original protocol unchanged (fpga.client). The only
addition is request 0x20, answered by the fpga_viz bitstream with the telemetry of the
inference that just finished. The payload carries that inference's frame ID, so a
telemetry block can never be paired with the wrong camera frame.
"""
import struct
import time
import zlib

from fpga.client import FPGAClient, MAGIC, preprocess, NAMES
from .telemetry import PAYLOAD_BYTES, decode_payload


class TelemetryUnavailable(RuntimeError):
    pass


class VizClient(FPGAClient):
    def receive(self):
        deadline = time.perf_counter() + self.timeout
        window = bytearray()
        while bytes(window) != MAGIC:
            window.extend(self._read(1, deadline))
            if len(window) > 4:
                del window[0]
        header = self._read(10, deadline)
        version, kind, uid, length = struct.unpack('<BBII', header)
        if version != 1 or length > PAYLOAD_BYTES:
            raise ValueError('Invalid FPGA packet header')
        payload = self._read(length, deadline)
        crc = struct.unpack('<I', self._read(4, deadline))[0]
        if zlib.crc32(header + payload) != crc:
            raise ValueError('FPGA response CRC mismatch')
        return kind, uid, payload

    def telemetry(self, inference_frame):
        try:
            kind, payload = self.request(0x20)
        except RuntimeError as error:
            if 'error code 3' in str(error):
                raise TelemetryUnavailable('The board runs the original bitstream (no telemetry). Run ProgramVizFPGA.cmd.') from error
            raise
        if kind != 0xA0:
            raise ValueError('Unexpected telemetry response')
        block = decode_payload(payload)
        if block['frame_id'] != inference_frame:
            raise ValueError(f"Telemetry is for frame {block['frame_id']}, expected {inference_frame}")
        return block


__all__ = ['VizClient', 'TelemetryUnavailable', 'preprocess', 'NAMES']


class VizDetectorClient:
    """Detector client with telemetry readback (fpga_viz/det_rtl bitstream). Detection packets are
    the original fpga_det protocol; request 0x20 returns the telemetry of the last inference."""
    def __new__(cls, port=None, timeout=3.):
        from fpga_det.client import DetectorClient
        from .det_telemetry import PAYLOAD_BYTES as DET_PAYLOAD, decode_payload as det_decode

        class _Client(DetectorClient):
            def receive(self):
                deadline = time.perf_counter() + self.timeout
                window = bytearray()
                while bytes(window) != MAGIC:
                    window.extend(self._read(1, deadline))
                    if len(window) > 4:
                        del window[0]
                header = self._read(10, deadline)
                version, kind, uid, length = struct.unpack('<BBII', header)
                if version != 1 or length > DET_PAYLOAD:
                    raise ValueError('Invalid FPGA packet header')
                payload = self._read(length, deadline)
                crc = struct.unpack('<I', self._read(4, deadline))[0]
                if zlib.crc32(header + payload) != crc:
                    raise ValueError('FPGA response CRC mismatch')
                return kind, uid, payload

            def telemetry(self, inference_frame):
                try:
                    kind, payload = self.request(0x20)
                except RuntimeError as error:
                    if 'error code 3' in str(error):
                        raise TelemetryUnavailable('The board runs the original detector bitstream (no telemetry). Run ProgramDetectorDemo.cmd.') from error
                    raise
                if kind != 0xA0:
                    raise ValueError('Unexpected telemetry response')
                block = det_decode(payload)
                if block['frame_id'] != inference_frame:
                    raise ValueError(f"Telemetry is for frame {block['frame_id']}, expected {inference_frame}")
                return block

        return _Client(port, timeout)
