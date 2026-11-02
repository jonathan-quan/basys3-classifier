"""XC7A35T chip map drawn from Vivado's REAL routed placement (fpga_viz/build/placement).

Every coloured tile is a tile that holds cells of the design; each functional group can be
made to glow. Nothing here is decorative guesswork: positions come from placed_cells.csv."""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# key, legend label, RGB colour. Order = classification priority (first regex match wins).
GROUPS = [
    ('bank_a', 'Activation bank A (8 BRAM)', (60, 190, 255)),
    ('bank_b', 'Activation bank B (8 BRAM)', (190, 110, 255)),
    ('telemetry', 'Telemetry tap (demo)', (150, 160, 255)),
    ('weights', 'Weight ROM (in LUTs)', (255, 150, 40)),
    ('params', 'Bias/scale ROM (BRAM)', (255, 105, 80)),
    ('mac', 'Multiply-accumulate', (255, 225, 60)),
    ('requant', 'Requantize + ReLU', (150, 255, 110)),
    ('head', 'Pool, FC, softmax', (255, 85, 175)),
    ('control', 'Control + addressing', (205, 215, 230)),
    ('uart', 'UART + packet protocol', (70, 235, 200)),
]
GROUP_KEYS = [g[0] for g in GROUPS]
PATTERNS = {
    'bank_a': lambda n, r: r.startswith('RAMB') and 'activation_a' in n,
    'bank_b': lambda n, r: r.startswith('RAMB') and 'activation_b' in n,
    'telemetry': lambda n, r: n.startswith('u_tel/'),
    'weights': lambda n, r: bool(re.search(r'p_0_out_inferred|weight_q|weight_addr|/weights', n)),
    'params': lambda n, r: bool(re.search(r'parameter_q|bias_reg|requant_multiplier|requant_shift|cfg_', n)),
    'head': lambda n, r: bool(re.search(r'pool|logits|fc_|exp_|dividend|remainder|quotient|denominator|divide_count|max_logit|result_|winner'
                                        r'|heat|att_|dot_|div_|size_out|presence|num_|exp_total|soft', n)),
    'mac': lambda n, r: bool(re.search(r'mac_|accumulator|tap_valid', n)),
    'requant': lambda n, r: bool(re.search(r'requant_product|rq_|write_data|saturate', n)),
    'control': lambda n, r: n.startswith('u_core/'),
    'uart': lambda n, r: True,
}
REGION_COLORS = {'X0Y2': (40, 150, 190), 'X1Y2': (230, 200, 50), 'X0Y1': (235, 135, 165),
                 'X1Y1': (80, 210, 165), 'X0Y0': (175, 225, 45), 'X1Y0': (140, 90, 230)}


def bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def classify_cell(name, ref):
    for key in GROUP_KEYS:
        if PATTERNS[key](name, ref):
            return key
    return 'uart'


class ChipMap:
    def __init__(self, placement_dir, width=560, height=840):
        placement_dir = Path(placement_dir)
        self.width, self.height = width, height
        tiles = list(csv.DictReader(open(placement_dir / 'device_tiles.csv', newline='')))
        cells = list(csv.DictReader(open(placement_dir / 'placed_cells.csv', newline='')))
        self.columns = max(int(t['column']) for t in tiles) + 1
        self.rows = max(int(t['row']) for t in tiles)
        self.cw = (width - 8) / self.columns
        self.rh = (height - 8) / self.rows
        self.tile_rects = {t['tile']: self._rect(t['type'], int(t['column']), int(t['row'])) for t in tiles}

        # Group membership of every placed cell, per tile.
        self.group_cells = Counter()
        per_tile = defaultdict(Counter)
        for c in cells:
            group = classify_cell(c['cell'], c['ref'])
            self.group_cells[group] += 1
            if c['tile'] not in self.tile_rects:
                self.tile_rects[c['tile']] = self._rect('CLB', int(c['column']), int(c['row']))
            weight = 24.0 if c['ref'].startswith(('RAMB', 'DSP')) else 1.0
            per_tile[c['tile']][group] += weight
        self.total_cells = len(cells)

        # Static background: dark die, tile outlines, clock regions, dim used cells.
        base = np.zeros((height, width, 3), np.float32)
        base[:] = (12, 9, 7)
        for t in tiles:
            x0, y0, x1, y1 = self.tile_rects[t['tile']]
            kind = t['type']
            shade = (46, 30, 20) if kind.startswith('CLB') else (70, 46, 26) if kind.startswith(('BRAM', 'DSP')) else (40, 40, 40)
            cv2.rectangle(base, (x0, y0), (x1, y1), shade, -1)
        self._draw_regions(base, tiles)
        masks = np.zeros((len(GROUPS), height, width), np.float32)
        for tile, groups in per_tile.items():
            x0, y0, x1, y1 = self.tile_rects[tile]
            total = sum(groups.values())
            for group, count in groups.items():
                fill = min(1.0, count / 10.0) * (count / total) ** 0.5
                masks[GROUP_KEYS.index(group), y0:y1 + 1, x0:x1 + 1] = np.maximum(masks[GROUP_KEYS.index(group), y0:y1 + 1, x0:x1 + 1], fill)
        colors = np.array([bgr(g[2]) for g in GROUPS], np.float32) / 255.0
        dim = np.tensordot(masks.transpose(1, 2, 0), colors, axes=1) * 135.0
        self.base = np.clip(base + dim, 0, 255)
        # Where the two activation banks physically are (8 block RAMs each).
        self.bank_tiles = {'bank_a': [], 'bank_b': []}
        for c in cells:
            group = classify_cell(c['cell'], c['ref'])
            if group in self.bank_tiles:
                self.bank_tiles[group].append(self.tile_rects[c['tile']])

        # Glow layers at half resolution: crisp core + soft halo per group.
        hw, hh = width // 2, height // 2
        stack = []
        for i in range(len(GROUPS)):
            small = cv2.resize(masks[i], (hw, hh), interpolation=cv2.INTER_AREA)
            halo = cv2.GaussianBlur(small, (0, 0), 5.0) * 2.2 + cv2.GaussianBlur(small, (0, 0), 1.2) * 1.3
            stack.append((halo[..., None] * colors[i][None, None, :]).reshape(-1))
        self.glow_stack = np.stack(stack).astype(np.float32)
        # Crisp per-tile core: only the pixels that hold design cells (sparse, ~10% of the die).
        used = masks.max(axis=0) > 0
        self.core_index = np.flatnonzero(used)
        flat = masks.reshape(len(GROUPS), -1)[:, self.core_index]
        self.core_values = (flat[:, :, None] * colors[:, None, :]).reshape(len(GROUPS), -1).astype(np.float32)
        self.half = (hw, hh)

    def _rect(self, kind, column, row):
        x = 4 + column * self.cw
        if kind.startswith(('BRAM', 'DSP')):
            y0, y1 = 4 + (row - 5) * self.rh, 4 + row * self.rh - 1
            return int(x), int(y0), int(x + self.cw * 1.6), int(y1)
        if 'IOB' in kind:
            return int(x), int(4 + (row - 1) * self.rh), int(x + self.cw * 0.9), int(4 + (row + 1) * self.rh - 2)
        return int(x), int(4 + (row - 1) * self.rh), int(x + self.cw * 0.82), int(4 + row * self.rh - 1)

    def _draw_regions(self, base, tiles):
        boxes = defaultdict(list)
        for t in tiles:
            if t['clock_region']:
                boxes[t['clock_region']].append(self.tile_rects[t['tile']])
        self.region_boxes = {}
        for name, rects in boxes.items():
            x0 = min(r[0] for r in rects) - 2; y0 = min(r[1] for r in rects) - 2
            x1 = max(r[2] for r in rects) + 2; y1 = max(r[3] for r in rects) + 2
            self.region_boxes[name] = (x0, y0, x1, y1)
            color = REGION_COLORS.get(name, (120, 120, 120))
            cv2.rectangle(base, (x0, y0), (x1, y1), tuple(v * 0.55 for v in bgr(color)), 1, cv2.LINE_AA)

    def render(self, intensity):
        """intensity: dict group -> 0..1. Returns uint8 BGR image of the chip."""
        weights = np.array([float(intensity.get(k, 0.0)) for k in GROUP_KEYS], np.float32)
        glow = (weights @ self.glow_stack).reshape(self.half[1], self.half[0], 3)
        glow = cv2.resize(glow, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = self.base + glow * 170.0
        flat = image.reshape(-1, 3)
        flat[self.core_index] += (weights @ self.core_values).reshape(-1, 3) * 150.0
        return np.clip(image, 0, 255).astype(np.uint8)
