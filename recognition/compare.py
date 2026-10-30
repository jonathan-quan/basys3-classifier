"""Compare original classifier and detector CLASSIFICATION ONLY on a webcam.

1: original, 2: detector, Tab/Space: switch, F: native/shared full-frame inputs,
Q/Escape: quit. No boxes, localization, presence scores, optimizers or training.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from classifier.common import choose_device, seed_everything, file_hash
from classifier.data import image_transform
from classifier.evaluate import load_checkpoint as load_classifier
from classifier.webcam import crop_roi, open_camera

PROJECT = Path(__file__).resolve().parents[1]
NAMES = ("Rubik's Cube", "Pen", "Book", "Computer Mouse")
TITLE = "Webcam recognition comparison - classification only"


def detector_candidates(explicit=None):
    if explicit:
        return [Path(explicit).resolve()]
    paths = [PROJECT / 'detection' / 'best_detector.pth']
    runs = PROJECT / 'detection' / 'runs'
    if runs.is_dir():
        available = list(runs.glob('*/best_detector.pth'))
        available.sort(key=lambda p: (not (p.parent / 'summary.json').is_file(), -p.stat().st_mtime))
        paths.extend(available)
    return paths


class ModelBank:
    """Load available checkpoints independently and keep their tensors read-only."""
    def __init__(self, device, classifier=None, detector=None):
        self.device = device
        self.models = {}
        self.metadata = {}
        self.unavailable = {}
        self.original_transform = image_transform(False)
        self.classification_weights_identical = None
        classifier = Path(classifier) if classifier else PROJECT / 'best_model.pth'
        try:
            model, checkpoint = load_classifier(classifier, device)
            if checkpoint.get('smoke_test'):
                raise ValueError('Synthetic fixture checkpoint is not a usable classifier')
            self.models['original'] = model.eval()
            historical = file_hash(classifier) == '98c66c324f34d66a1c4d876ec3fcf05f6ed1f57f0df1a7c6733870fbb3e4cf61'
            self.metadata['original'] = {
                'name': 'Original classifier', 'checkpoint': str(classifier.resolve()),
                'status': ('Historical public-data test accuracy: 76.53%; live accuracy unmeasured'
                           if historical else 'Classifier loaded; live accuracy unmeasured'),
                'epoch': checkpoint.get('epoch'),
            }
        except Exception as error:
            self.unavailable['original'] = f'{type(error).__name__}: {error}'

        errors = []
        for path in detector_candidates(detector):
            if not path.is_file():
                continue
            try:
                # Optional imports: a broken/missing detector never disables the original.
                from detection.model import load_detector
                from detection.data import letterbox, tensorize
                model, checkpoint = load_detector(path, device)
                if checkpoint.get('smoke_test'):
                    raise ValueError('Synthetic fixture checkpoint is not usable')
                if not callable(getattr(model, 'classify', None)):
                    raise ValueError('Detector has no classification-only method')
                completed = (path.parent / 'summary.json').is_file()
                tested = any(p.is_file() for p in (path.parent / 'test_metrics.json',
                                                   path.parent / 'evaluation' / 'test_metrics.json'))
                if checkpoint.get('incomplete_pilot'):
                    status = 'INCOMPLETE PILOT checkpoint; recognition quality unvalidated'
                elif not completed:
                    status = 'INCOMPLETE training checkpoint; provisional classification only'
                elif not tested:
                    status = 'Training run finished; experimental checkpoint; detector test pending'
                else:
                    status = 'Trained detector class head; live recognition accuracy unmeasured'
                self.models['detector'] = model.eval()
                self.detector_transform = (letterbox, tensorize)
                self.metadata['detector'] = {
                    'name': 'Detector classification head', 'checkpoint': str(path.resolve()),
                    'epoch': checkpoint.get('epoch'), 'status': status,
                    'training_finished': completed, 'held_out_detector_test_found': tested,
                }
                break
            except Exception as error:
                errors.append(f'{path.name}: {type(error).__name__}: {error}')
        if 'detector' not in self.models:
            self.unavailable['detector'] = '; '.join(errors) if errors else (
                f'Checkpoint not found: {detector}' if detector else 'No compatible trained detector checkpoint found')
        if all(key in self.models for key in ('original', 'detector')):
            old = self.models['original'].state_dict()
            new = self.models['detector'].state_dict()
            self.classification_weights_identical = all(
                key in new and torch.equal(value, new[key]) for key, value in old.items())

    def predict(self, frame, key, shared_full_frame=False):
        if key not in self.models:
            raise ValueError(self.unavailable.get(key, 'Model unavailable'))
        source = frame
        if key == 'original' and not shared_full_frame:
            source, _ = crop_roi(frame)  # Exact original webcam's 75% central-square crop.
        image = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        if key == 'original':
            tensor = self.original_transform(image)
        else:
            letterbox, tensorize = self.detector_transform
            tensor = tensorize(letterbox(image)[0])  # Exact detector preprocessing.
        tensor = tensor.unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            if key == 'original':
                logits = self.models[key](tensor)
            else:
                # Deliberately bypass forward(): box and presence branches never execute.
                logits = self.models[key].classify(tensor)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite classification logits')
            scores = logits.softmax(dim=1)[0].cpu().tolist()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        prediction = int(np.argmax(scores))
        return {'model': key, 'class_id': prediction, 'name': NAMES[prediction],
                'class_score': scores[prediction], 'all_class_scores': scores,
                'model_ms': elapsed * 1000, 'shared_full_frame': shared_full_frame}

    def status(self):
        return {'available': self.metadata, 'unavailable': self.unavailable,
                'classification_weights_identical': self.classification_weights_identical,
                'score_definition': 'Maximum four-class softmax, NOT presence or joint detection confidence',
                'training_performed': False, 'bounding_boxes_used': False}


class PredictionWorker:
    def __init__(self, bank):
        self.bank = bank
        self.active = 'original' if 'original' in bank.models else next(iter(bank.models))
        self.shared_full_frame = False
        self.condition = threading.Condition()
        self.generation = 0
        self.pending = None
        self.result = None
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def select(self, key):
        if key not in self.bank.models:
            return False
        with self.condition:
            self.active = key
            self.generation += 1
            self.pending = self.result = self.error = None
        return True

    def toggle_framing(self):
        with self.condition:
            self.shared_full_frame = not self.shared_full_frame
            self.generation += 1
            self.pending = self.result = self.error = None

    def submit(self, frame, frame_id, captured_at):
        with self.condition:
            self.pending = (frame.copy(), frame_id, captured_at, self.active,
                            self.shared_full_frame, self.generation)
            self.condition.notify()

    def snapshot(self):
        with self.condition:
            return self.active, self.shared_full_frame, self.result, self.error

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                frame, uid, timestamp, key, shared, generation = self.pending
                self.pending = None
            try:
                result = self.bank.predict(frame, key, shared)
                result.update(frame_id=uid, captured_at=timestamp)
                with self.condition:
                    if generation == self.generation:
                        self.result, self.error = result, None
            except Exception as error:
                with self.condition:
                    if generation == self.generation:
                        self.error = f'{type(error).__name__}: {error}'
                        self.result = None

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=5)


def put_text(frame, text, y, color=(255, 255, 255), scale=.53):
    # OpenCV 5 can change glyph spacing with thickness. Make the outline from
    # the exact foreground mask, avoiding two differently spaced text passes.
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    mask = np.zeros((height + baseline + 6, width + 6), dtype=np.uint8)
    cv2.putText(mask, text, (3, height + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    outline = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    left, top = 9, y - height - 3
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(frame.shape[1], left + mask.shape[1]), min(frame.shape[0], top + mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    foreground = mask[y1-top:y2-top, x1-left:x2-left, None].astype(np.float32) / 255
    shadow = outline[y1-top:y2-top, x1-left:x2-left, None].astype(np.float32) / 255
    region = frame[y1:y2, x1:x2].astype(np.float32) * (1 - shadow)
    frame[y1:y2, x1:x2] = (region * (1 - foreground) + np.array(color) * foreground).astype(np.uint8)


def render(frame, bank, active, shared, result=None, error=None, notice='', preview_fps=0.):
    # Text only. Do not add object boxes, fixed ROI rectangles or coordinate overlays.
    metadata = bank.metadata[active]
    put_text(frame, f"ACTIVE: {metadata['name']}", 28, (60, 240, 100), .65)
    fresh = result and time.perf_counter() - result.get('captured_at', time.perf_counter()) < 1
    if fresh:
        put_text(frame, f"{result['name']}: {result['class_score']:.1%} class score", 63, (60, 240, 100), .8)
    else:
        put_text(frame, 'Prediction unavailable' if error else 'Waiting for prediction...', 63, (50, 180, 255), .65)
    framing = 'Full frame for BOTH models' if shared else (
        'Original framing: central 75% square; hold object near center'
        if active == 'original' else 'Detector framing: full frame, exact 96x96 letterbox')
    put_text(frame, framing, 91)
    put_text(frame, '1 Original | 2 Detector | Tab/Space switch | F framing | Q quit', 116, scale=.48)
    latency = f" | Model {result['model_ms']:.1f} ms" if fresh else ''
    put_text(frame, f"LAPTOP ONLY ({bank.device}) | Preview {preview_fps:.1f} FPS{latency}", 141, scale=.48)
    put_text(frame, 'Uncalibrated class scores; every frame is assigned one of four classes', 166, scale=.45)
    if active == 'detector':
        put_text(frame, f"Epoch {metadata['epoch']} | {metadata['status']}", 191, (50, 180, 255), .43)
        if bank.classification_weights_identical:
            put_text(frame, 'Classification weights match original; framing can change predictions', 213, (50, 180, 255), .43)
    elif 'detector' not in bank.models:
        put_text(frame, 'Detector unavailable; original classifier works independently', 191, (50, 180, 255), .45)
    if error or notice:
        put_text(frame, (error or notice)[:100], max(240, frame.shape[0] - 15), (50, 180, 255), .43)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier', default=str(PROJECT / 'best_model.pth'))
    parser.add_argument('--detector', help='Optional exact detector checkpoint; otherwise use the published checkpoint or latest available run')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--image', help='Camera-free functional check; does not measure webcam accuracy')
    parser.add_argument('--output', help='Optional text-only image preview for the camera-free check')
    args = parser.parse_args()
    seed_everything(42)
    bank = ModelBank(choose_device(args.device), args.classifier, args.detector)
    print(json.dumps(bank.status(), indent=2), flush=True)
    if not bank.models:
        raise RuntimeError('Neither model could be loaded. See the checkpoint errors above.')
    if args.image:
        with Image.open(args.image) as image:
            frame = cv2.cvtColor(np.array(ImageOps.exif_transpose(image).convert('RGB')), cv2.COLOR_RGB2BGR)
        for key in bank.models:
            result = bank.predict(frame, key)
            print(json.dumps(result), flush=True)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path = path.with_name(path.stem + '_' + key + path.suffix)
                # Show controls/status at readable resolution; model used the original image.
                preview = cv2.resize(frame, (960, 720))
                if not cv2.imwrite(str(path), render(preview, bank, key, False, result)):
                    raise OSError(f'Cannot save {path}')
        return
    print('Focus the video window. 1=original, 2=detector, Tab/Space=switch, F=framing, Q/Escape=quit.', flush=True)
    camera = open_camera(args.camera)
    worker = PredictionWorker(bank)
    started = time.perf_counter()
    frames = 0
    notice = ''
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames. Close other camera apps or try camera 1.')
            now = time.perf_counter()
            frames += 1
            worker.submit(frame, frames, now)
            active, shared, result, error = worker.snapshot()
            render(frame, bank, active, shared, result, error, notice, frames / max(.001, now - started))
            cv2.imshow(TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            requested = None
            if key == ord('1'):
                requested = 'original'
            elif key == ord('2'):
                requested = 'detector'
            elif key in (9, 32):
                requested = 'detector' if active == 'original' else 'original'
            elif key in (ord('f'), ord('F')):
                worker.toggle_framing()
            if requested:
                if worker.select(requested):
                    notice = ''
                else:
                    notice = requested.capitalize() + ' unavailable: ' + bank.unavailable.get(requested, '')
            if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        worker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
