"""Recording view: laptop webcam + Basys 3 FPGA (box detector or classifier) + live chip map.

Frame-locked: every frame the FPGA classifies is shown together with THAT frame's own
result and telemetry. The whole view therefore runs a fixed delay (~0.55 s) behind real
time (the FPGA round trip is ~0.36 s); on screen, the object appears, the chip lights up
for that exact frame, and its answer follows with the real upload + compute latency.

Keys: R record MP4 · SPACE freeze + slow replay of one inference · F centre square / full
frame · H help · Q or Esc quit
"""
import argparse
import collections
import multiprocessing
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import ROOT, PROJECT
from .chipmap import ChipMap
from .client import VizClient, TelemetryUnavailable, preprocess, NAMES
from .profiles import classifier as classifier_profile, detector as detector_profile
from .scene import Scene, W, H
from .telemetry import TOTAL_CYCLES, CLOCK_HZ, reference_block, blocks_equal
WINDOW = 'Basys 3 FPGA - recording view'


@dataclass
class Transaction:
    t_capture: float
    image_sent: np.ndarray       # exactly the RGB bytes the FPGA received
    result: dict
    telemetry: object
    t_done: float
    crop: tuple
    box_frame: object = None     # detector: box in camera-frame pixels
    verified: object = None


class Camera:
    def __init__(self, index=0, image=None, video=None):
        self.frames = collections.deque(maxlen=150)
        self.lock = threading.Lock()
        self.running = True
        self.still = None
        self.cap = None
        self.size = None
        if image:
            self.still = cv2.imread(str(image))
            if self.still is None:
                raise RuntimeError(f'Cannot read {image}')
        elif video:
            self.cap = cv2.VideoCapture(str(video))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        else:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError('Cannot open webcam; close other camera apps and check Windows camera permissions')
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.fps = None
        self.video = video
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            if self.still is not None:
                frame = self.still.copy()
                time.sleep(1 / 30)
            else:
                ok, frame = self.cap.read()
                if not ok:
                    if self.video:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                if self.video:
                    time.sleep(1 / self.fps)
            stamp = time.perf_counter()
            self.size = frame.shape[1::-1]
            with self.lock:
                self.frames.append((stamp, frame))

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None

    def at(self, t):
        with self.lock:
            for stamp, frame in reversed(self.frames):
                if stamp <= t:
                    return stamp, frame
            return self.frames[0] if self.frames else None

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()


class SimDetector:
    """Stand-in for testing without a board: the detector's integer reference. Clearly labelled."""
    def __init__(self):
        from fpga_det.quantize import IntegerDetector
        from fpga_det.client import RESULT, decode_result
        self.reference = IntegerDetector.load(PROJECT / 'fpga_det' / 'export')
        self.RESULT, self.decode = RESULT, decode_result
        self.frame_id = 0

    def detect_letterboxed(self, letterboxed):
        start = time.perf_counter()
        q = (np.asarray(letterboxed, dtype=np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
        out = self.reference.run(q, trace=True)
        self.last = out
        cls = int(np.argmax(out['class_logits'][0]))
        packed = self.RESULT.pack(cls, 0, *[int(v) for v in out['class_logits'][0]], int(out['presence'][0]), int(out['cx16'][0]),
                                  int(out['cy16'][0]), *[int(v) for v in out['size'][0]], 20118456)
        result = self.decode(packed)
        self.frame_id += 1
        time.sleep(max(0.0, 0.70 - (time.perf_counter() - start)))
        result.update(roundtrip_ms=(time.perf_counter() - start) * 1000, frame_id=self.frame_id, inference_location='SIMULATION')
        return result

    def telemetry(self, frame_id):
        from .det_telemetry import reference_block as det_block
        time.sleep(0.07)
        block = det_block(self.last)
        block['frame_id'] = frame_id
        return block

    def close(self):
        pass


class SimClient:
    """Stand-in for testing without a board: the laptop integer reference. Clearly labelled."""
    def __init__(self):
        from fpga.quantize import IntegerClassifier, classify_logits
        self.reference = IntegerClassifier.load(PROJECT / 'fpga' / 'export')
        self.classify_logits = classify_logits
        self.frame_id = 0
        self.last = None

    def classify(self, image):
        start = time.perf_counter()
        pixels = np.array(preprocess(image))
        q = (pixels.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
        logits, states, pooled = self.reference.run(q, trace=True)
        label, confidence = self.classify_logits(logits[0])
        self.last = (states, pooled[0])
        self.frame_id += 1
        time.sleep(max(0.0, 0.31 - (time.perf_counter() - start)))
        return {'class_id': label, 'name': NAMES[label], 'class_score': confidence / 65535, 'confidence_u16': confidence,
                'logits_q8_8': tuple(int(v) for v in logits[0]), 'inference_cycles': TOTAL_CYCLES,
                'inference_ms': TOTAL_CYCLES / 100000, 'roundtrip_ms': (time.perf_counter() - start) * 1000,
                'frame_id': self.frame_id, 'inference_location': 'SIMULATION'}

    def telemetry(self, frame_id):
        time.sleep(0.03)
        block = reference_block(*self.last)
        block['frame_id'] = frame_id
        return block

    def close(self):
        pass


def _verify_loop(model, inbox, outbox):
    """Runs in a separate process so the reference check never stalls the video."""
    if model == 'detector':
        from fpga_det.quantize import IntegerDetector
        from fpga_viz.det_telemetry import reference_block as det_block, blocks_equal as det_equal
        reference = IntegerDetector.load(PROJECT / 'fpga_det' / 'export')
    else:
        from fpga.quantize import IntegerClassifier
        from fpga_viz.telemetry import reference_block as ref_block, blocks_equal as equal
        reference = IntegerClassifier.load(PROJECT / 'fpga' / 'export')
    while True:
        item = inbox.get()
        if item is None:
            break
        key, image, fpga_output, telemetry = item
        if model == 'detector':
            q = (np.asarray(image, dtype=np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
            out = reference.run(q, trace=True)
            raw = {'class_logits_q8_8': [int(v) for v in out['class_logits'][0]], 'presence_q8_8': int(out['presence'][0]),
                   'cx_q16': int(out['cx16'][0]), 'cy_q16': int(out['cy16'][0]), 'size_q8_8': [int(v) for v in out['size'][0]]}
            ok = raw == fpga_output
            if ok and telemetry is not None:
                ok = det_equal(det_block(out), telemetry)
        else:
            q = (image.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)
            logits, states, pooled = reference.run(q, trace=True)
            ok = list(logits[0]) == list(fpga_output)
            if telemetry is not None:
                ok = ok and equal(ref_block(states, pooled[0]), telemetry)
        outbox.put((key, bool(ok)))


class Verifier:
    """Re-runs the integer reference on the exact 96x96 bytes each frame sent and checks that the
    FPGA's logits AND telemetry are bit-identical, which also proves the frame pairing is right."""
    def __init__(self, model):
        self.model = model
        context = multiprocessing.get_context('spawn')
        self.inbox = context.Queue(maxsize=16)
        self.outbox = context.Queue()
        self.process = context.Process(target=_verify_loop, args=(model, self.inbox, self.outbox), daemon=True)
        self.process.start()
        self.pending = {}
        self.checked = self.matched = 0
        threading.Thread(target=self._collect, daemon=True).start()

    def submit(self, tx):
        key = id(tx)
        self.pending[key] = tx
        try:
            output = tx.result['raw'] if self.model == 'detector' else list(tx.result['logits_q8_8'])
            self.inbox.put_nowait((key, tx.image_sent, output, tx.telemetry))
        except queue.Full:
            self.pending.pop(key, None)

    def _collect(self):
        while True:
            key, ok = self.outbox.get()
            tx = self.pending.pop(key, None)
            if tx is not None:
                tx.verified = ok
            self.checked += 1
            self.matched += int(ok)

    def close(self):
        try:
            self.inbox.put_nowait(None)
        except Exception:
            pass


class Worker:
    def __init__(self, camera, port, simulate, verifier, model):
        self.camera, self.port, self.simulate, self.verifier, self.model = camera, port, simulate, verifier, model
        self.transactions = collections.deque(maxlen=64)
        self.lock = threading.Lock()
        self.inflight = None
        self.client = None
        self.error = 'Connecting to FPGA…'
        self.notice = None
        self.telemetry_ok = None
        self.full_frame = False
        self.running = True
        self.count = 0
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        last = 0.0
        while self.running:
            try:
                if self.client is None:
                    if self.model == 'detector':
                        from .client import VizDetectorClient
                        self.client = SimDetector() if self.simulate else VizDetectorClient(self.port)
                    else:
                        self.client = SimClient() if self.simulate else VizClient(self.port)
                    self.error = None
                item = self.camera.latest()
                if item is None or item[0] <= last:
                    time.sleep(0.002)
                    continue
                stamp, frame = item
                last = stamp
                self.inflight = stamp
                h, w = frame.shape[:2]
                if self.model == 'detector':
                    tx = self._detect(stamp, frame)
                    with self.lock:
                        self.transactions.append(tx)
                    self.inflight = None
                    self.count += 1
                    if self.verifier is not None:
                        self.verifier.submit(tx)
                    continue
                if self.full_frame:
                    crop = (0, 0, w, h)
                    region = frame
                else:
                    side = int(min(h, w) * .75)
                    crop = ((w - side) // 2, (h - side) // 2, side, side)
                    region = frame[crop[1]:crop[1] + side, crop[0]:crop[0] + side]
                image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                sent = np.array(preprocess(image))          # exactly the bytes classify() transmits
                result = self.client.classify(image)
                telemetry = None
                if self.telemetry_ok is not False:
                    try:
                        telemetry = self.client.telemetry(result['frame_id'])
                        self.telemetry_ok = True
                    except TelemetryUnavailable as error:
                        self.telemetry_ok = False
                        self.notice = str(error)
                tx = Transaction(stamp, sent, result, telemetry, time.perf_counter(), crop)
                with self.lock:
                    self.transactions.append(tx)
                self.inflight = None
                self.count += 1
                if self.verifier is not None:
                    self.verifier.submit(tx)
            except Exception as error:  # keep the view alive; reconnect
                self.error = str(error)[:120]
                self.inflight = None
                if self.client is not None:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                    self.client = None
                time.sleep(1.0)
        if self.client is not None:
            self.client.close()

    def _detect(self, stamp, frame):
        from detection.improve import letterbox
        from fpga_det.client import SIDE
        from fpga_det.webcam import frame_box
        h, w = frame.shape[:2]
        letterboxed, _ = letterbox(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), SIDE)
        result = self.client.detect_letterboxed(letterboxed)
        result['frame_size'] = (w, h)
        telemetry = None
        if self.telemetry_ok is not False:
            try:
                telemetry = self.client.telemetry(result['frame_id'])
                self.telemetry_ok = True
            except TelemetryUnavailable as error:
                self.telemetry_ok = False
                self.notice = str(error)
        return Transaction(stamp, np.asarray(letterboxed, dtype=np.uint8).copy(), result, telemetry, time.perf_counter(),
                           (0, 0, w, h), frame_box(result))

    def history(self):
        with self.lock:
            return list(self.transactions)

    def close(self):
        self.running = False
        self.thread.join(timeout=3)


class Recorder:
    """Writes the composed view at a constant 30 fps. Uses ffmpeg (H.264, CRF 18) when it is
    installed, which keeps small text sharp; otherwise falls back to OpenCV's MPEG-4 writer."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.queue = queue.Queue(maxsize=90)
        self.recording = False
        self.finished = threading.Event()
        self.finished.set()
        self.path = None
        self.started = 0.0
        self.written = 0
        self.ffmpeg = shutil.which('ffmpeg')
        threading.Thread(target=self.run, daemon=True).start()

    @property
    def active(self):
        return self.recording

    def toggle(self):
        if not self.recording:
            self.finished.wait(timeout=30)          # previous file fully written
            self.folder.mkdir(parents=True, exist_ok=True)
            self.path = self.folder / f"fpga_demo_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            self.finished.clear()
            self.queue.put(('open', self.path))
            self.started = time.perf_counter()
            self.written = 0
            self.recording = True
        else:
            self.recording = False
            self.queue.put(('close', None))

    def push(self, frame):
        if not self.recording:
            return
        due = int((time.perf_counter() - self.started) * 30) + 1     # constant 30 fps timeline
        while self.written < due:
            try:
                self.queue.put_nowait(('frame', frame))
            except queue.Full:
                break
            self.written += 1

    def run(self):
        process = writer = None
        while True:
            kind, item = self.queue.get()
            if kind == 'open':
                if self.ffmpeg:
                    process = subprocess.Popen(
                        [self.ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                         '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(item)],
                        stdin=subprocess.PIPE, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                else:
                    writer = cv2.VideoWriter(str(item), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
            elif kind == 'frame':
                if process is not None:
                    process.stdin.write(item.tobytes())
                elif writer is not None:
                    writer.write(item)
            elif kind == 'close':
                if process is not None:
                    process.stdin.close()
                    process.wait(timeout=120)
                    process = None
                if writer is not None:
                    writer.release()
                    writer = None
                print('SAVED', self.path, flush=True)
                self.finished.set()

    def close(self):
        if self.recording:
            self.toggle()
        self.finished.wait(timeout=60)


def pick(history, t, answer_offset=0.0):
    """current = newest transaction whose frame time <= t; answered = newest whose answer time <= t.

    The recording view uses answer_offset=0: the view is already delayed by more than the FPGA round
    trip, so each answer is known before its frame is displayed and is drawn starting ON that frame.
    (A positive offset would draw it on later frames, so boxes would trail moving objects.)"""
    current = answered = previous = None
    for i in range(len(history) - 1, -1, -1):
        tx = history[i]
        if current is None and tx.t_capture <= t:
            current = tx
            previous = history[i - 1] if i > 0 else None
        if answered is None and tx.t_capture + answer_offset <= t:
            answered = tx
        if current is not None and answered is not None:
            break
    return current, answered, previous


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=('detector', 'classifier'), default='detector',
                   help='detector: class + box (fpga_det bitstream); classifier: 4 classes + telemetry maps (fpga_viz bitstream)')
    p.add_argument('--port')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--image', help='use a still image instead of the webcam (testing)')
    p.add_argument('--video', help='use a video file instead of the webcam (testing)')
    p.add_argument('--simulate', action='store_true', help='no board: laptop integer reference, clearly labelled on screen')
    p.add_argument('--delay', type=float, help='display delay in seconds (must exceed the FPGA round trip; default per model)')
    p.add_argument('--no-verify', action='store_true')
    p.add_argument('--placement', help='placement export folder (default per model)')
    p.add_argument('--recordings', default=str(PROJECT / 'recordings'))
    p.add_argument('--fullscreen', action='store_true')
    p.add_argument('--snapshot', help='render N seconds, save the last frame to this PNG and exit (testing)')
    p.add_argument('--seconds', type=float, default=6.0)
    p.add_argument('--test-replay', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--test-record', type=float, default=0, help=argparse.SUPPRESS)
    p.add_argument('--answer-offset', type=float, default=0.0, help=argparse.SUPPRESS)  # testing only; 0 = answer drawn on its own frame
    a = p.parse_args()

    profile = detector_profile() if a.model == 'detector' else classifier_profile()
    placement = Path(a.placement) if a.placement else ROOT / 'build' / profile.placement
    if not (placement / 'placed_cells.csv').exists():
        placement = ROOT / 'build' / 'placement_original'
    chip = ChipMap(placement)
    scene = Scene(chip, profile)
    camera = Camera(a.camera, a.image, a.video)
    verifier = None if (a.no_verify or a.simulate) else Verifier(a.model)
    worker = Worker(camera, a.port, a.simulate, verifier, a.model)
    recorder = Recorder(a.recordings)
    delay = a.delay if a.delay else profile.delay
    compute_s = profile.total_cycles / CLOCK_HZ
    slow_replay = None          # (transaction, wall-clock start) while frozen
    show_help = False
    late_events = 0
    seen = set()
    if not a.snapshot:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if a.fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW, 1600, 900)
    started = time.perf_counter()
    last_frame = None
    frame_times = []
    if a.test_record:
        recorder.toggle()
    try:
        while True:
            loop_start = time.perf_counter()
            now = loop_start
            t = now - delay
            history = worker.history()
            if worker.inflight is not None and worker.inflight <= t:
                late_events += 1           # round trip longer than the delay: lengthen it
                if late_events % 15 == 0:
                    delay = min(1.5, delay + 0.05)
            if slow_replay is None:
                intake, sweep, outro = profile.live_timing
                current, answered, previous = pick(history, t, a.answer_offset)
                for tx in history:
                    if id(tx) not in seen and tx.t_capture <= t:
                        seen.add(id(tx))
                        scene.note_transaction(tx)
                age = (t - current.t_capture) if current is not None else -1
                item = camera.at(t)
                frame = item[1] if item is not None else None
            else:
                current, started_replay = slow_replay
                intake, sweep, outro = profile.replay_timing
                answered, previous = current, None
                age = (now - started_replay) % (intake + sweep + outro + 1.2)
                frame = slow_replay_frame
            status = {}
            if a.simulate:
                status['banner'] = 'SIMULATION — laptop reference, not the FPGA'
            elif worker.error:
                status['banner'] = worker.error[:70]
            elif worker.telemetry_ok is False and not a.simulate:
                status['banner'] = ('Original bitstream: run ProgramVizFPGA.cmd for activation maps' if a.model == 'classifier'
                                    else 'Original bitstream: run ProgramDetectorDemo.cmd for activation maps')
            elapsed = max(0.01, now - worker.started)
            parts = []
            if answered is not None:
                rt = answered.result['roundtrip_ms']
                parts.append(f"upload + compute {rt:.0f} ms")
            parts.append(f"{worker.count / elapsed:.1f} FPGA updates/s")
            if verifier is not None and verifier.checked:
                parts.append(f"bit-exact vs reference {verifier.matched}/{verifier.checked}")
            if slow_replay is not None:
                status['paused'] = f'PAUSED  ·  replaying this frame\'s {profile.compute_ms:.0f} ms inference {sweep / compute_s:.0f}× slower  ·  SPACE resumes'
            status['footer'] = '  ·  '.join(parts)
            canvas = scene.render(frame, current, answered, previous, age, (sweep, intake, outro), status)
            last_frame = canvas
            recorder.push(canvas)
            if a.snapshot:
                frame_times.append((time.perf_counter() - loop_start) * 1000)
                if a.test_replay and slow_replay is None and now - started > a.seconds * 0.6 and current is not None:
                    slow_replay = (current, time.perf_counter())
                    slow_replay_frame = frame
                if now - started > a.seconds:
                    cv2.imwrite(a.snapshot, canvas)
                    ft = np.array(frame_times[30:]) if len(frame_times) > 40 else np.array(frame_times)
                    print('SNAPSHOT', a.snapshot, 'transactions', worker.count, 'error', worker.error,
                          'fps', round(len(frame_times) / (now - started), 1),
                          'render_ms p50/p95/max', np.round(np.percentile(ft, [50, 95, 100]), 1).tolist(),
                          'frames>50ms', int((ft > 50).sum()), 'of', len(ft),
                          'verified', None if verifier is None else (verifier.matched, verifier.checked), flush=True)
                    break
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - loop_start)))
                continue
            view = canvas
            if recorder.active or show_help:
                view = canvas.copy()
                if recorder.active:
                    cv2.circle(view, (W - 40, 30), 11, (40, 40, 230), -1, cv2.LINE_AA)
                    scene.text.draw(view, f'REC {time.perf_counter() - recorder.started:5.1f}s', W - 60, 18, 20, (255, 90, 90), 'bold', align='right')
                if show_help:
                    help_lines = ['R  record MP4 (saved in recordings\\)', 'SPACE  freeze + slow replay of one inference',
                                  'A  attention overlay on/off (detector)', 'F  centre square / full frame (classifier)',
                                  'H  hide this help', 'Q / Esc  quit',
                                  f'display delay {delay:.2f} s (keeps every frame paired with its own result)']
                    cv2.rectangle(view, (W // 2 - 330, H // 2 - 150), (W // 2 + 330, H // 2 + 160), (20, 16, 12), -1)
                    for i, line in enumerate(help_lines):
                        scene.text.draw(view, line, W // 2 - 300, H // 2 - 110 + i * 38, 22)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                recorder.toggle()
            if key in (ord('h'), ord('H')):
                show_help = not show_help
            if key in (ord('f'), ord('F')):
                worker.full_frame = not worker.full_frame
            if key in (ord('a'), ord('A')):
                scene.show_attention = not scene.show_attention
            if key == ord(' '):
                if slow_replay is None and current is not None:
                    slow_replay = (current, time.perf_counter())
                    slow_replay_frame = frame
                else:
                    slow_replay = None
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if a.test_record and now - started > a.test_record:
                break
            spare = 1 / 30 - (time.perf_counter() - loop_start)
            if spare > 0.002:
                time.sleep(spare - 0.001)
    finally:
        recorder.close()
        worker.close()
        camera.close()
        if verifier is not None:
            verifier.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
