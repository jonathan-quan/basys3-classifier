"""Draws the 1920x1080 recording view: camera, FPGA result, chip map, activity maps."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .chipmap import GROUPS, GROUP_KEYS, bgr

W, H = 1920, 1080
CHIP = (1010, 72, 560, 840)
SIDE_X, SIDE_W = 1596, 296
MAPS_Y = 930
ACCENT = (70, 235, 200)
INK = (225, 232, 240)
MUTED = (120, 132, 150)
CLASS_COLORS = [(60, 190, 255), (255, 150, 40), (150, 255, 110), (255, 85, 175)]
FONT_DIR = Path('C:/Windows/Fonts')


class Text:
    def __init__(self):
        self.cache = {}
        self.blend = {}
        self.fonts = {}

    def font(self, size, weight):
        key = (size, weight)
        if key not in self.fonts:
            names = {'regular': ['segoeui.ttf', 'arial.ttf'], 'bold': ['segoeuib.ttf', 'arialbd.ttf'],
                     'light': ['segoeuil.ttf', 'segoeui.ttf'], 'mono': ['consola.ttf', 'cour.ttf']}[weight]
            font = None
            for name in names:
                try:
                    font = ImageFont.truetype(str(FONT_DIR / name), size)
                    break
                except OSError:
                    continue
            self.fonts[key] = font or ImageFont.load_default()
        return self.fonts[key]

    def patch(self, text, size, weight):
        key = (text, size, weight)
        if key not in self.cache:
            if len(self.cache) > 3000:
                self.cache.clear()
            font = self.font(size, weight)
            left, top, right, bottom = font.getbbox(text)
            image = Image.new('L', (max(1, right - left + 2), max(1, bottom - top + 2)), 0)
            ImageDraw.Draw(image).text((1 - left, 1 - top), text, font=font, fill=255)
            self.cache[key] = (np.asarray(image, np.float32) / 255.0, top)
        return self.cache[key]

    def draw(self, canvas, text, x, y, size=20, color=INK, weight='regular', alpha=1.0, align='left'):
        """y is the top of the em box; returns the text width (alpha=0 only measures)."""
        if not text:
            return 0
        mask, top = self.patch(text, size, weight)
        h, w = mask.shape
        if align == 'right':
            x -= w
        elif align == 'center':
            x -= w // 2
        if alpha <= 0:
            return w
        x, y = int(x), int(y + top)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return w
        key = (text, size, weight, tuple(color), round(alpha, 3))
        layers = self.blend.get(key)
        if layers is None:
            if len(self.blend) > 3000:
                self.blend.clear()
            m = (mask * alpha).astype(np.float32)
            colored = np.empty((h, w, 3), np.uint8)
            colored[:] = bgr(color)
            layers = self.blend[key] = (colored, np.ascontiguousarray(1.0 - m), np.ascontiguousarray(m))
        colored, w0, w1 = layers
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            sy, sx = slice(y0 - y, y1 - y), slice(x0 - x, x1 - x)
            colored, w0, w1 = (np.ascontiguousarray(a[sy, sx]) for a in (colored, w0, w1))
        canvas[y0:y1, x0:x1] = cv2.blendLinear(np.ascontiguousarray(canvas[y0:y1, x0:x1]), colored, w0, w1)
        return w


def heat_color(values):
    v = np.clip(values, 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


class Scene:
    def __init__(self, chip, profile, chip_title='XC7A35T  ·  real placement from Vivado'):
        self.chip = chip
        self.profile = profile
        self.text = Text()
        aw, ah = profile.camera_aspect
        self.cam = (40, 72, 940, 940 * ah // aw)
        self.background = self._background(chip_title)
        n = len(profile.layer_names)
        self.layer_ends = list(profile.layer_ends)
        self.layer_starts = [0] + self.layer_ends[:-1]
        self.peak_heat = [1.0] * n
        self.peak_channel = [1.0] * n
        self.peak_activity = [1e-6] * n
        self.smoothed = {k: 0.0 for k in GROUP_KEYS}
        self.show_attention = True
        self._attention_cache = (None, None)      # (transaction id, (x0, y0, x1, y1, colored, w_frame, w_overlay))
        self._tile_cache = {}

    # ---- static parts ------------------------------------------------------------
    def _background(self, chip_title):
        bg = np.zeros((H, W, 3), np.uint8)
        bg[:] = (14, 11, 8)
        for x in range(0, W, 40):
            cv2.line(bg, (x, 0), (x, H), (24, 20, 16), 1)
        for y in range(0, H, 40):
            cv2.line(bg, (0, y), (W, y), (24, 20, 16), 1)
        t = Text()
        t.draw(bg, self.profile.title, self.cam[0], 12, 26, INK, 'bold')
        t.draw(bg, self.profile.subtitle, self.cam[0], 47, 15, MUTED)
        t.draw(bg, 'THE CHIP', CHIP[0], 12, 26, INK, 'bold')
        t.draw(bg, chip_title, CHIP[0], 47, 15, MUTED)
        cv2.rectangle(bg, (CHIP[0] - 1, CHIP[1] - 1), (CHIP[0] + CHIP[2], CHIP[1] + CHIP[3]), (60, 48, 36), 1)
        return bg

    # ---- helpers -----------------------------------------------------------------
    def phase(self, age, sweep, intake, outro):
        if age < 0:
            return 'idle', 0.0, None
        if age < intake:
            return 'input', age / intake, None
        a = age - intake
        if a < sweep:
            cycle = a / sweep * self.profile.total_cycles
            for layer, end in enumerate(self.layer_ends):
                if cycle < end:
                    return 'layer', cycle, layer
            return 'head', cycle, None
        a -= sweep
        if a < outro:
            return 'output', a / outro, None
        return 'hold', a, None

    def activity(self, tx):
        n = len(self.profile.layer_names)
        if tx is None or tx.telemetry is None:
            return [0.5] * n
        return [float(np.sum(sums)) / (self.profile.layer_cout[i] * self.profile.layer_hw[i] ** 2 * 127.0)
                for i, sums in enumerate(tx.telemetry['channel_sums'])]

    def intensities(self, kind, progress, layer, tx):
        level = {k: 0.05 for k in GROUP_KEYS}
        act = self.activity(tx)
        if kind == 'input':
            level.update(uart=1.0, bank_a=0.35 + 0.65 * progress, control=0.2)
        elif kind == 'layer':
            src, dst = ('bank_a', 'bank_b') if layer % 2 == 0 else ('bank_b', 'bank_a')
            relative = act[layer] / max(self.peak_activity[layer], 1e-6) if tx is not None and tx.telemetry is not None else 0.7
            level.update(mac=1.0, weights=0.95, params=0.55, control=0.7, requant=0.85)
            level[src] = 0.7
            level[dst] = 0.3 + 0.7 * min(1.0, relative)
            level['telemetry'] = 0.6 if tx is not None and tx.telemetry is not None else 0.05
        elif kind == 'head':
            level.update(bank_a=0.6, bank_b=0.8, head=1.0, weights=0.5, params=0.5, control=0.5, mac=0.4)
        elif kind == 'output':
            level.update(uart=1.0, head=0.8 * (1 - progress) + 0.2)
        elif kind == 'hold':
            level.update(head=0.18, uart=0.1)
        for k, v in level.items():
            s = self.smoothed[k]
            self.smoothed[k] = v if v > s else s * 0.72 + v * 0.28
        return dict(self.smoothed)

    def note_transaction(self, tx):
        if tx is None or tx.telemetry is None:
            return
        for i, a in enumerate(self.activity(tx)):
            self.peak_activity[i] = max(self.peak_activity[i] * 0.97, a)
        for i, heat in enumerate(tx.telemetry['heat']):
            self.peak_heat[i] = max(self.peak_heat[i] * 0.97, float(heat.max()), 1.0)
        for i, sums in enumerate(tx.telemetry['channel_sums']):
            self.peak_channel[i] = max(self.peak_channel[i] * 0.97, float(sums.max()), 1.0)

    # ---- frame ---------------------------------------------------------------------
    def render(self, camera_frame, current, answered, previous, age, timing, status):
        canvas = self.background.copy()
        kind, progress, layer = self.phase(age, *timing) if current is not None else ('idle', 0.0, None)
        self._camera(canvas, camera_frame, current, answered, status)
        self._input_and_classes(canvas, current, answered, status)
        levels = self.intensities(kind, progress, layer, current)
        canvas[CHIP[1]:CHIP[1] + CHIP[3], CHIP[0]:CHIP[0] + CHIP[2]] = self.chip.render(levels)
        self._region_labels(canvas)
        self._side(canvas, kind, progress, layer, current, levels, timing, status)
        self._maps(canvas, kind, progress, layer, current, previous)
        return canvas

    def _view_geometry(self, frame):
        """Crop of the camera frame shown in the panel and the frame->canvas mapping."""
        x, y, w, h = self.cam
        fh, fw = frame.shape[:2]
        if fw * h > fh * w:
            cw = fh * w // h; ox, oy = (fw - cw) // 2, 0; view = frame[:, ox:ox + cw]
        else:
            ch = fw * h // w; ox, oy = 0, (fh - ch) // 2; view = frame[oy:oy + ch, :]
        scale = w / view.shape[1]
        return view, ox, oy, scale

    @staticmethod
    def _attention_grid(tx):
        """Detector head attention (16x16 softmax weights from the FPGA) on the 128x128 letterbox grid, 0..1."""
        if tx is None or tx.telemetry is None or 'attention' not in tx.telemetry:
            return None
        att = tx.telemetry['attention'].astype(np.float32)
        att = (att / max(float(att.max()), 1.0)) ** 2.5   # display gamma: emphasise the peak the box centre comes from
        return cv2.resize(att, (128, 128), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _heat_layers(heat, strength=0.55, floor=0.12):
        colored = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        alpha = (np.clip((heat - floor) / (1 - floor), 0, 1) * strength).astype(np.float32)
        return colored, np.ascontiguousarray(1.0 - alpha), np.ascontiguousarray(alpha)

    @classmethod
    def _blend_heat(cls, roi, heat, strength=0.55, floor=0.12):
        colored, w0, w1 = cls._heat_layers(heat, strength, floor)
        return cv2.blendLinear(np.ascontiguousarray(roi), colored, w0, w1)

    def _camera_attention(self, canvas, tx, ox, oy, scale):
        x, y, w, h = self.cam
        key = (id(tx), ox, oy, round(scale, 6))
        if self._attention_cache[0] == key:
            cached = self._attention_cache[1]
            if cached is not None:
                x0, y0, x1, y1, colored, w0, w1 = cached
                roi = np.ascontiguousarray(canvas[y + y0:y + y1, x + x0:x + x1])
                canvas[y + y0:y + y1, x + x0:x + x1] = cv2.blendLinear(roi, colored, w0, w1)
            return
        self._attention_cache = (key, None)
        grid = self._attention_grid(tx)
        if grid is None:
            return
        fw, fh = tx.result['frame_size']
        s = min(128 / fw, 128 / fh); nw, nh = round(fw * s), round(fh * s); left, top = (128 - nw) // 2, (128 - nh) // 2
        region = grid[top:top + nh, left:left + nw]
        pw, ph = int(round(fw * scale)), int(round(fh * scale))
        overlay = cv2.resize(region, (pw, ph), interpolation=cv2.INTER_LINEAR)
        dx, dy = int(round(-ox * scale)), int(round(-oy * scale))
        x0, y0, x1, y1 = max(0, dx), max(0, dy), min(w, dx + pw), min(h, dy + ph)
        if x1 <= x0 or y1 <= y0:
            return
        sub = overlay[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        colored, w0, w1 = self._heat_layers(sub, 0.6, 0.06)
        self._attention_cache = (key, (x0, y0, x1, y1, colored, w0, w1))
        roi = np.ascontiguousarray(canvas[y + y0:y + y1, x + x0:x + x1])
        canvas[y + y0:y + y1, x + x0:x + x1] = cv2.blendLinear(roi, colored, w0, w1)

    def _camera(self, canvas, frame, current, answered, status):
        x, y, w, h = self.cam
        if frame is None:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 34, 28), 1)
            self.text.draw(canvas, 'Waiting for camera…', x + w // 2, y + h // 2 - 12, 24, MUTED, align='center')
            return
        view, ox, oy, scale = self._view_geometry(frame)
        canvas[y:y + h, x:x + w] = cv2.resize(view, (w, h), interpolation=cv2.INTER_LINEAR)

        def to_canvas(px, py):
            return int(np.clip(x + (px - ox) * scale, x, x + w - 1)), int(np.clip(y + (py - oy) * scale, y, y + h - 1))

        detector = self.profile.key == 'detector'
        # The attention glow is drawn on the camera only when an object is detected; with no object the
        # softmax still peaks somewhere, which reads as a false hot spot (the input thumbnail keeps it).
        if (detector and self.show_attention and answered is not None
                and answered.result['score'] >= self.profile.threshold):
            self._camera_attention(canvas, answered, ox, oy, scale)
        if current is not None and not detector:
            cx, cy, cw_, ch_ = current.crop
            x0, y0 = to_canvas(cx, cy); x1, y1 = to_canvas(cx + cw_, cy + ch_)
            corner, color = 34, bgr(ACCENT)
            for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                cv2.line(canvas, (px, py), (px + dx * corner, py), color, 3, cv2.LINE_AA)
                cv2.line(canvas, (px, py), (px, py + dy * corner), color, 3, cv2.LINE_AA)
            self.text.draw(canvas, 'FPGA field of view', x0 + 10, y1 - 28, 16, ACCENT)
        if answered is not None:
            r = answered.result
            color = CLASS_COLORS[r['class_id']]
            found = (not detector) or r['score'] >= self.profile.threshold
            panel = canvas[y + 18:y + 128, x + 18:x + 560].astype(np.float32)
            canvas[y + 18:y + 128, x + 18:x + 560] = (panel * 0.35).astype(np.uint8)
            if found:
                cv2.rectangle(canvas, (x + 18, y + 18), (x + 24, y + 128), bgr(color), -1)
                self.text.draw(canvas, r['name'].upper(), x + 40, y + 26, 54, color, 'bold')
                if detector:
                    detail = f"score {r['score']:.0%} = presence {r['presence']:.0%} × class {r['class_score']:.0%}  ·  {r['inference_ms']:.0f} ms on chip"
                else:
                    detail = f"FPGA score {r['class_score']:.0%}  ·  computed in {r['inference_ms']:.2f} ms on chip"
            else:
                cv2.rectangle(canvas, (x + 18, y + 18), (x + 24, y + 128), (90, 80, 70), -1)
                self.text.draw(canvas, 'NO OBJECT', x + 40, y + 26, 54, MUTED, 'bold')
                detail = f"best guess {r['name']} {r['score']:.0%} < {self.profile.threshold:.0%} threshold  ·  {r['inference_ms']:.0f} ms on chip"
            self.text.draw(canvas, detail, x + 42, y + 96, 18, INK)
            if detector and found:
                bx0, by0, bx1, by1 = answered.box_frame
                x0, y0 = to_canvas(bx0, by0); x1, y1 = to_canvas(bx1, by1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), bgr(color), 3, cv2.LINE_AA)
                tag = f"{r['name'].upper()}  {r['score']:.0%}"
                tw = self.text.draw(canvas, tag, x0 + 8, y0, 20, color, 'bold', alpha=0.0)
                above = y0 - 34
                hits_panel = above < y + 132 and x0 < x + 564
                ty = above if (above > y and not hits_panel) else min(y1 + 4, y + h - 34)
                tx0 = min(x0, x + w - tw - 18)
                cv2.rectangle(canvas, (tx0, ty), (tx0 + tw + 16, ty + 30), bgr(color), -1)
                self.text.draw(canvas, tag, tx0 + 8, ty + 4, 20, (12, 14, 18), 'bold')
        if status.get('paused'):
            bw = self.text.draw(canvas, status['paused'], x + 22, y + h - 70, 18, INK, 'bold', alpha=0.0)
            panel = canvas[y + h - 78:y + h - 44, x + 12:x + 34 + bw].astype(np.float32)
            canvas[y + h - 78:y + h - 44, x + 12:x + 34 + bw] = (panel * 0.3).astype(np.uint8)
            self.text.draw(canvas, status['paused'], x + 22, y + h - 70, 18, (255, 225, 60), 'bold')
        if status.get('banner'):
            bw = self.text.draw(canvas, status['banner'], x + w - 14, y + h - 34, 18, (255, 120, 90), 'bold', align='right', alpha=0.0)
            panel = canvas[y + h - 42:y + h - 8, x + w - bw - 26:x + w - 6].astype(np.float32)
            canvas[y + h - 42:y + h - 8, x + w - bw - 26:x + w - 6] = (panel * 0.3).astype(np.uint8)
            self.text.draw(canvas, status['banner'], x + w - 14, y + h - 34, 18, (255, 120, 90), 'bold', align='right')

    def _input_and_classes(self, canvas, current, answered, status):
        x, y = self.cam[0], self.cam[1] + self.cam[3] + 16
        side = self.profile.input_side
        size = int(min(256, 1060 - y - 30))
        label = f'FPGA INPUT  ·  {side}×{side}'
        self.text.draw(canvas, label, x, y, 15, MUTED, 'bold')
        box = canvas[y + 26:y + 26 + size, x:x + size]
        if current is not None:
            thumb = cv2.resize(cv2.cvtColor(current.image_sent, cv2.COLOR_RGB2BGR), (size, size), interpolation=cv2.INTER_NEAREST)
            r = current.result
            grid = self._attention_grid(current) if (self.profile.key == 'detector' and current is answered) else None
            if grid is not None:
                thumb = self._blend_heat(thumb, cv2.resize(grid, (size, size), interpolation=cv2.INTER_LINEAR), 0.65, 0.06)
            if self.profile.key == 'detector' and current is answered and r['score'] >= self.profile.threshold:
                cx, cy, bw, bh = r['box_cxcywh_letterbox']
                cv2.rectangle(thumb, (int((cx - bw / 2) * size), int((cy - bh / 2) * size)),
                              (int((cx + bw / 2) * size), int((cy + bh / 2) * size)), bgr(CLASS_COLORS[r['class_id']]), 2)
            box[:] = thumb
        cv2.rectangle(canvas, (x - 1, y + 25), (x + size, y + 26 + size), bgr(ACCENT), 1)
        if self.profile.key == 'detector':
            self.text.draw(canvas, 'glow = where the FPGA\'s box head looked', x, y + 34 + size, 14, MUTED)
        bx = x + size + 40
        detector = self.profile.key == 'detector'
        title = 'OUTPUT NEURONS  ·  class + presence logits from the FPGA' if detector else 'OUTPUT NEURONS  ·  4 logits from the FPGA'
        self.text.draw(canvas, title, bx, y, 15, MUTED, 'bold')
        if answered is None:
            return
        r = answered.result
        if detector:
            logits = np.array(r['raw']['class_logits_q8_8'], np.float64) / 256.0
        else:
            logits = np.array(r['logits_q8_8'], np.float64) / 256.0
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        winner = r['class_id']
        rows = [(self.profile.names[i], probs[i], CLASS_COLORS[i], i == winner) for i in range(4)]
        if detector:
            rows.append(('Object present', r['presence'], (230, 230, 230), r['presence'] >= 0.5))
        bar_x, bar_w = bx + 200, max(160, 980 - (bx + 200) - 60)
        for i, (name, value, color, on) in enumerate(rows):
            ry = y + 32 + i * 46
            cv2.circle(canvas, (bx + 12, ry + 16), 10, bgr(color) if on else (70, 60, 50), -1 if on else 2, cv2.LINE_AA)
            self.text.draw(canvas, name, bx + 34, ry + 3, 21, INK if on else MUTED, 'bold' if on else 'regular')
            cv2.rectangle(canvas, (bar_x, ry + 8), (bar_x + bar_w, ry + 26), (38, 32, 26), -1)
            fill = int(bar_w * value)
            if fill > 0:
                cv2.rectangle(canvas, (bar_x, ry + 8), (bar_x + fill, ry + 26), bgr(color) if on else tuple(int(c * 0.45) for c in bgr(color)), -1)
            self.text.draw(canvas, f'{value:.0%}', bar_x + bar_w + 10, ry + 4, 19, INK if on else MUTED, 'mono')
        self.text.draw(canvas, status.get('footer', ''), bx, y + 32 + len(rows) * 46 + 8, 15, MUTED)

    def _region_labels(self, canvas):
        from .chipmap import REGION_COLORS
        for group, letter in (('bank_a', 'A'), ('bank_b', 'B')):
            color = GROUPS[GROUP_KEYS.index(group)][2]
            for (x0, y0, x1, y1) in self.chip.bank_tiles.get(group, []):
                self.text.draw(canvas, letter, CHIP[0] + x1 + 2, CHIP[1] + (y0 + y1) // 2 - 7, 12, color, 'bold', alpha=0.9)
        for name, (x0, y0, x1, y1) in self.chip.region_boxes.items():
            self.text.draw(canvas, name, CHIP[0] + x0 + 5, CHIP[1] + y1 - 24, 17, tuple(int(c * 0.8) for c in REGION_COLORS.get(name, MUTED)), 'bold', alpha=0.8)

    def _side(self, canvas, kind, progress, layer, tx, levels, timing, status):
        prof = self.profile
        n = len(prof.layer_names)
        x, y = SIDE_X, CHIP[1]
        self.text.draw(canvas, 'NOW ON THE CHIP', x, y, 15, MUTED, 'bold')
        if kind == 'input':
            headline, detail = 'Receiving image', f'{prof.input_side * prof.input_side * 3:,} bytes over UART → bank A'
        elif kind == 'layer':
            headline = f'Layer {layer + 1} of {n}'
            detail = f'{prof.layer_names[layer]}  ·  {prof.layer_cout[layer]} ch  ·  {prof.layer_hw[layer]}×{prof.layer_hw[layer]}'
        elif kind == 'head':
            headline, detail = prof.head_name, prof.head_detail
        elif kind == 'output':
            headline, detail = 'Sending answer', 'result packet' + (' + telemetry' if tx is not None and tx.telemetry is not None else '')
        else:
            headline, detail = ('Idle' if tx is None else 'Waiting for next frame'), ''
        self.text.draw(canvas, headline, x, y + 24, 30, INK, 'bold')
        self.text.draw(canvas, detail, x, y + 64, 16, ACCENT)
        compute_s = prof.total_cycles / 100_000_000
        if kind in ('layer', 'head') and timing[0] > compute_s * 1.3:
            self.text.draw(canvas, f'{prof.compute_ms:.2f} ms on chip · shown {timing[0] / compute_s:.1f}× slower'.replace('.0×', '×'), x, y + 88, 14, MUTED)
        ly = y + 122
        self.text.draw(canvas, 'WHAT EACH COLOUR IS', x, ly, 15, MUTED, 'bold')
        for i, (key, label, color) in enumerate(GROUPS):
            ry = ly + 26 + i * 28
            lvl = levels.get(key, 0.0)
            c = tuple(int(v * (0.35 + 0.65 * min(1, lvl * 1.2))) for v in bgr(color))
            cv2.rectangle(canvas, (x, ry + 4), (x + 14, ry + 18), c, -1)
            if key in ('bank_a', 'bank_b'):
                label = f"Activation bank {'A' if key == 'bank_a' else 'B'} ({self.chip.group_cells.get(key, 0)} BRAM)"
            self.text.draw(canvas, label, x + 24, ry, 16, INK if lvl > 0.3 else MUTED)
            self.text.draw(canvas, f'{self.chip.group_cells.get(key, 0):,}', x + SIDE_W, ry + 2, 14, MUTED, 'mono', align='right')
        ty = ly + 26 + len(GROUPS) * 28 + 14
        self.text.draw(canvas, f'WHERE THE {prof.compute_ms:.2f} ms GOES' if prof.compute_ms < 100 else f'WHERE THE {prof.compute_ms:.0f} ms GOES', x, ty, 15, MUTED, 'bold')
        cycle = progress if kind in ('layer', 'head') else (prof.total_cycles if kind in ('output', 'hold') else 0)
        stamps = tx.telemetry['stamps'] if tx is not None and tx.telemetry is not None else self.layer_ends
        starts = [0] + list(stamps[:-1])
        row_h = 25 if n <= 9 else 22
        biggest = max(s - t for s, t in zip(stamps, starts)) / prof.total_cycles
        for i in range(n):
            ry = ty + 26 + i * row_h
            share = (stamps[i] - starts[i]) / prof.total_cycles
            active = kind == 'layer' and layer == i
            done = cycle >= stamps[i]
            self.text.draw(canvas, f'L{i + 1}', x, ry, 15, INK if active else MUTED, 'bold' if active else 'regular')
            bar_w = SIDE_W - 92
            cv2.rectangle(canvas, (x + 34, ry + 5), (x + 34 + bar_w, ry + 16), (34, 28, 22), -1)
            fill = max(2, int(bar_w * share / biggest))
            col = bgr(GROUPS[5][2]) if active else ((120, 100, 70) if done else (60, 52, 44))
            cv2.rectangle(canvas, (x + 34, ry + 5), (x + 34 + fill, ry + 16), col, -1)
            self.text.draw(canvas, f'{share * 100:4.1f}%', x + SIDE_W, ry + 1, 14, INK if active else MUTED, 'mono', align='right')

    def _maps(self, canvas, kind, progress, layer, current, previous):
        prof = self.profile
        n = len(prof.layer_names)
        x0, y = CHIP[0], MAPS_Y
        width = W - 30 - x0
        gap = 14
        size = min(84, (width - gap * (n - 1)) // n)
        has = (current is not None and current.telemetry is not None) or (previous is not None and previous.telemetry is not None)
        title = 'ACTIVATION MAPS READ BACK FROM THE FPGA  ·  one per layer, summed over channels'
        if not has:
            title = 'ACTIVATION MAPS  ·  need the telemetry bitstream for this design'
        self.text.draw(canvas, title, x0, y - 6, 15, MUTED, 'bold')
        reached = layer if kind == 'layer' else (n if kind in ('head', 'output', 'hold') else -1)
        for i in range(n):
            mx = x0 + i * (size + gap)
            my = y + 18
            source, alpha = None, 1.0
            if current is not None and current.telemetry is not None and i <= reached:
                source = current
                if i == reached and kind == 'layer':
                    span = max(1, self.layer_ends[i] - self.layer_starts[i])
                    alpha = 0.6 + 0.4 * min(1.0, (progress - self.layer_starts[i]) / span)
            elif previous is not None and previous.telemetry is not None:
                source, alpha = previous, 0.35
            cv2.rectangle(canvas, (mx - 1, my - 1), (mx + size, my + size), (60, 48, 36), 1)
            if source is not None:
                key = (id(source), i, size)
                if key not in self._tile_cache:
                    if len(self._tile_cache) > 400:
                        self._tile_cache.clear()
                    heat = source.telemetry['heat'][i].astype(np.float32) / self.peak_heat[i]
                    sums = source.telemetry['channel_sums'][i].astype(np.float32) / self.peak_channel[i]
                    self._tile_cache[key] = (cv2.resize(heat_color(heat), (size, size), interpolation=cv2.INTER_NEAREST),
                                             cv2.resize(heat_color(sums[None, :]), (size, 9), interpolation=cv2.INTER_NEAREST))
                tile, strip = self._tile_cache[key]
                canvas[my:my + size, mx:mx + size] = tile if alpha >= 0.999 else cv2.convertScaleAbs(tile, alpha=alpha)
                canvas[my + size + 4:my + size + 13, mx:mx + size] = strip if alpha >= 0.999 else cv2.convertScaleAbs(strip, alpha=alpha)
            active = kind == 'layer' and layer == i
            self.text.draw(canvas, f'L{i + 1}', mx, my + size + 16, 14, INK if active else MUTED, 'bold' if active else 'regular')
            self.text.draw(canvas, f'{prof.layer_cout[i]}ch', mx + size, my + size + 16, 13, MUTED, align='right')
