"""Improved single-object detector experiments (software only).

This is NEW, separate experiment code. It does not modify the original detector
code, data, manifests, checkpoints, or runs. Every run writes into a new folder
under detection/runs/ and refuses to overwrite an existing one.

Selection uses VALIDATION only. The test split is evaluated exactly once, by the
separate --final-test command, for a run chosen beforehand from validation.

Changes versus detection/train.py (reviewed_cosine):
  * soft-argmax box head: a center heat map on the stride-8 feature grid
    (12x12 at 96 px) plus a size regressor, instead of one linear layer on the
    flattened 6x6 map ("fc" head kept as an option for a like-for-like control);
  * zoom-crop augmentation from the original image (objects seen at many scales,
    including the tiny webcam cubes enlarged), wider affine/colour jitter;
  * L1 + GIoU box loss, auxiliary center-cell loss, EMA weights, no frozen warm-up;
  * optional larger input (128) and wider from-scratch backbone for comparison.
"""
import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from classifier.common import choose_device, environment, file_hash, seed_everything
from classifier.data import read_manifest
from . import CLASSES, PROJECT, ROOT
from .data import TRUSTED, validate_rows
from .fetch_data import read_jsonl, save_json
from .metrics import compute, select_thresholds

CACHE_SIDE = 256  # training-only cache of the EXIF-corrected original (for zoom crops)

EXPERIMENTS = {
    # name: channels, context blocks, input size, head, transfer original classifier, training BN
    "r96_fc": dict(channels=(8, 16, 32, 48), context=1, size=96, head="fc", transfer=True, bn=False),
    "r96_sa": dict(channels=(8, 16, 32, 48), context=1, size=96, head="softargmax", transfer=True, bn=False),
    "r128_sa": dict(channels=(8, 16, 32, 48), context=1, size=128, head="softargmax", transfer=True, bn=False),
    "w128_sa": dict(channels=(16, 32, 64, 96), context=2, size=128, head="softargmax", transfer=False, bn=True),
}


# ----------------------------------------------------------------------------- data
def letterbox(image, size, box=None):
    """Same geometry as detection.data.letterbox, with a configurable size."""
    w, h = image.size
    scale = min(size / w, size / h)
    nw = max(1, round(w * scale)); nh = max(1, round(h * scale))
    left = (size - nw) // 2; top = (size - nh) // 2
    out = Image.new("RGB", (size, size), (128, 128, 128))
    out.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    b = None if box is None else np.array([(box[0] * nw + left) / size, (box[1] * nh + top) / size,
                                           (box[2] * nw + left) / size, (box[3] * nh + top) / size], dtype=np.float32)
    return out, b


def tensorize(image):
    a = np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)
    return torch.from_numpy(a).sub_(128).div_(128)


def zoom_crop(image, box):
    """Random crop of the original image that fully contains the object."""
    w, h = image.size
    if box is None:
        f = random.uniform(.5, 1.)
        cw = max(8, int(w * f)); ch = max(8, int(h * f))
        x0 = random.randint(0, w - cw); y0 = random.randint(0, h - ch)
        return image.crop((x0, y0, x0 + cw, y0 + ch)), None
    bx0, by0, bx1, by1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    bw = max(1., bx1 - bx0); bh = max(1., by1 - by0)
    f = random.uniform(1.25, 3.5)
    side = max(bw, bh) * f  # roughly square context window around the object
    cw = min(w, max(side, bw + 2)); ch = min(h, max(side, bh + 2))
    x0 = random.uniform(max(0., bx1 - cw), min(bx0, w - cw))
    y0 = random.uniform(max(0., by1 - ch), min(by0, h - ch))
    x0i, y0i = int(math.floor(x0)), int(math.floor(y0))
    x1i, y1i = min(w, int(math.ceil(x0 + cw))), min(h, int(math.ceil(y0 + ch)))
    cw, ch = x1i - x0i, y1i - y0i
    nb = np.clip([(bx0 - x0i) / cw, (by0 - y0i) / ch, (bx1 - x0i) / cw, (by1 - y0i) / ch], 0, 1)
    if nb[2] - nb[0] <= 0 or nb[3] - nb[1] <= 0:
        return image, box
    return image.crop((x0i, y0i, x1i, y1i)), nb.astype(np.float32)


def augment(image, box, size):
    """Affine + flip + colour, transforming boxes with pixels (as detection.data.augment)."""
    angle = random.uniform(-15, 15); scale = random.uniform(.80, 1.15)
    tx = random.uniform(-.10, .10) * size; ty = random.uniform(-.10, .10) * size
    theta = np.deg2rad(angle); c = np.cos(theta) * scale; s = np.sin(theta) * scale
    forward = np.array([[c, -s, size / 2 + tx - c * size / 2 + s * size / 2],
                        [s, c, size / 2 + ty - s * size / 2 - c * size / 2], [0, 0, 1]])
    new_box = None
    if box is not None:
        x1, y1, x2, y2 = np.asarray(box) * size
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ forward.T
        low = pts[:, :2].min(0); high = pts[:, :2].max(0)
        clipped = np.clip(np.r_[low, high], 0, size) / size
        visible = float(np.prod((clipped[2:] - clipped[:2]) * size)) / max(1e-6, float(np.prod(high - low)))
        if visible < .65 or np.any((clipped[2:] - clipped[:2]) * size < 1):
            forward = np.eye(3); new_box = np.array(box, dtype=np.float32, copy=True)
        else:
            new_box = clipped.astype(np.float32)
    inv = np.linalg.inv(forward)
    image = image.transform((size, size), Image.Transform.AFFINE, tuple(inv[:2].reshape(-1)),
                            resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    if random.random() < .5:
        image = ImageOps.mirror(image)
        if new_box is not None:
            new_box = np.array([1 - new_box[2], new_box[1], 1 - new_box[0], new_box[3]], dtype=np.float32)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(.6, 1.4))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(.7, 1.3))
    image = ImageEnhance.Color(image).enhance(random.uniform(.7, 1.3))
    if random.random() < .15:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(.2, 1.0)))
    return image, new_box


class DetectionSet(Dataset):
    def __init__(self, rows, split, size, training=False, zoom=.5, limit=None):
        self.rows = [r for r in rows if r.get("split") == split][:limit]
        self.size = size; self.training = training; self.zoom = zoom; self.items = []
        for r in self.rows:
            if r.get("annotation_status") not in TRUSTED:
                raise ValueError(f"Unverified candidate cannot be used: {r['id']}")
            if file_hash(r["path"]) != r["sha256"]:
                raise ValueError("Image changed after dataset preparation")
            with Image.open(r["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                if training:
                    im.thumbnail((CACHE_SIDE, CACHE_SIDE), Image.Resampling.BILINEAR)
                    self.items.append((im, r.get("bbox")))
                else:  # deterministic evaluation directly from the original image
                    self.items.append(letterbox(im, size, r.get("bbox")))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.training:
            image, box = self.items[i]
            box = None if box is None else np.asarray(box, dtype=np.float32)
            if random.random() < self.zoom:
                image, box = zoom_crop(image, box)
            image, box = letterbox(image, self.size, box)
            image, box = augment(image, box, self.size)
        else:
            image, box = self.items[i]
        return {"image": tensorize(image), "label": row["label"], "present": float(row["present"]),
                "box": torch.tensor(box if box is not None else [0, 0, 0, 0], dtype=torch.float32), "index": i}


class RehearsalSet(Dataset):
    """Original classifier TRAIN images only (class labels, no boxes are invented)."""
    def __init__(self, size, limit=None):
        rows = [r for r in read_manifest(PROJECT / "data/manifest.csv") if r["split"] == "train"][:limit]
        self.labels = [CLASSES.index(r["label"]) for r in rows]; self.size = size; self.items = []
        for r in rows:
            with Image.open(r["path"]) as im:
                self.items.append(letterbox(ImageOps.exif_transpose(im).convert("RGB"), size)[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image, _ = augment(self.items[i], None, self.size)
        return tensorize(image), self.labels[i]


# ---------------------------------------------------------------------------- model
def build_features(channels, context, bn):
    """Same layout as classifier.model.TinyClassifier('context') when channels/context match."""
    specs = [(3, channels[0], 3, 2, 1, 1)]; prev = channels[0]; tap_conv = None
    for i, c in enumerate(channels[1:]):
        specs += [(prev, prev, 3, 2, 1, prev), (prev, c, 1, 1, 0, 1)]; prev = c
        if i == 1:
            tap_conv = len(specs) - 1  # stride-8 map, after the second stage
    for _ in range(context):
        specs += [(prev, prev, 3, 1, 1, prev), (prev, prev, 1, 1, 0, 1)]
    layers = []; tap = None
    for k, (ci, co, ks, st, pd, g) in enumerate(specs):
        conv = nn.Conv2d(ci, co, ks, st, pd, groups=g, bias=not bn)
        nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        layers.append(conv)
        if bn:
            layers.append(nn.BatchNorm2d(co))
        layers.append(nn.ReLU())
        if k == tap_conv:
            tap = len(layers) - 1
    return nn.Sequential(*layers), tap


class Detector(nn.Module):
    def __init__(self, channels=(8, 16, 32, 48), context=1, size=96, head="softargmax", bn=False, **_):
        super().__init__()
        self.cfg = dict(channels=tuple(channels), context=context, size=size, head=head)
        self.features, self.tap = build_features(channels, context, bn)
        cf, ct = channels[-1], channels[2]
        self.fc = nn.Linear(cf, 4); self.presence = nn.Linear(cf, 1)
        nn.init.zeros_(self.presence.weight); nn.init.zeros_(self.presence.bias)
        self.grid = size // 8
        if head == "softargmax":
            self.heat_fine = nn.Conv2d(ct, 1, 1); self.heat_coarse = nn.Conv2d(cf, 1, 1)
            self.size_fc = nn.Linear(cf + ct, 2)
            for m in (self.heat_fine, self.heat_coarse, self.size_fc):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            nn.init.constant_(self.size_fc.bias, -0.4)  # sigmoid ~0.4 initial size
        elif head == "fc":
            g = size // 16
            self.box_reduce = nn.Conv2d(cf, 8, 1); self.box_fc = nn.Linear(8 * g * g, 4)
            nn.init.kaiming_normal_(self.box_reduce.weight, nonlinearity="relu"); nn.init.zeros_(self.box_reduce.bias)
            nn.init.normal_(self.box_fc.weight, std=.002); nn.init.zeros_(self.box_fc.bias)
        else:
            raise ValueError(head)

    def backbone(self, x):
        tap = None
        for i, m in enumerate(self.features):
            x = m(x)
            if i == self.tap:
                tap = x
        return x, tap

    def classify(self, x):
        return self.fc(self.backbone(x)[0].mean((2, 3)))

    def forward(self, images):
        if images.shape[-1] != self.cfg["size"] or images.shape[-2] != self.cfg["size"]:
            raise ValueError("Wrong input size")
        final, tap = self.backbone(images)
        pooled = final.mean((2, 3))
        out = {"class_logits": self.fc(pooled), "presence_logits": self.presence(pooled).squeeze(1)}
        if self.cfg["head"] == "softargmax":
            heat = self.heat_fine(tap) + F.interpolate(self.heat_coarse(final), size=tap.shape[-2:], mode="nearest")
            b, _, gh, gw = heat.shape
            logits = heat.flatten(1); prob = logits.softmax(1)
            ys = (torch.arange(gh, device=heat.device, dtype=heat.dtype) + .5) / gh
            xs = (torch.arange(gw, device=heat.device, dtype=heat.dtype) + .5) / gw
            cx = (prob * xs.repeat(gh)).sum(1); cy = (prob * ys.repeat_interleave(gw)).sum(1)
            attended = (prob.view(b, 1, gh, gw) * tap).sum((2, 3))
            wh = self.size_fc(torch.cat([pooled, attended], 1)).sigmoid()
            out["box_cxcywh"] = torch.stack([cx, cy, wh[:, 0], wh[:, 1]], 1)
            out["heat_logits"] = logits
        else:
            out["box_cxcywh"] = self.box_fc(F.relu(self.box_reduce(final)).flatten(1)).sigmoid()
        return out


def transfer_original(model, path):
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c.get("architecture") != "context" or c.get("classes") != list(CLASSES):
        raise ValueError("Expected the original 6,084-parameter context classifier")
    state = c["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected transfer keys {result.unexpected_keys}")
    assert sum(v.numel() for v in state.values()) == 6084
    return {"checkpoint_sha256": file_hash(path), "parameters_transferred": 6084}


def fold_for_inference(model):
    """Fold training BN into convolutions; returns a BN-free Detector with identical outputs."""
    cfg = model.cfg; plain = Detector(**cfg, bn=False)
    trained = model.state_dict(); target = plain.state_dict()
    for key, value in trained.items():  # heads share names
        if not key.startswith("features.") and key in target:
            target[key] = value.detach().clone()
    convs, bns = [], []
    mods = list(model.features)
    for i, m in enumerate(mods):
        if isinstance(m, nn.Conv2d):
            bn = mods[i + 1] if i + 1 < len(mods) and isinstance(mods[i + 1], nn.BatchNorm2d) else None
            convs.append(torch.nn.utils.fusion.fuse_conv_bn_eval(copy.deepcopy(m).eval(), bn.eval()) if bn is not None else m)
    plain_convs = [m for m in plain.features if isinstance(m, nn.Conv2d)]
    assert len(plain_convs) == len(convs)
    plain.load_state_dict(target)
    with torch.no_grad():
        for p, c in zip(plain_convs, convs):
            p.weight.copy_(c.weight); p.bias.copy_(c.bias)
    return plain.eval()


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def macs(model):
    total = []; hooks = []
    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            total.append(out.numel() // out.shape[0] * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1])
        else:
            total.append(m.in_features * m.out_features)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        model.eval()(torch.zeros(1, 3, model.cfg["size"], model.cfg["size"]))
    for h in hooks:
        h.remove()
    return int(sum(total))


# ----------------------------------------------------------------------- training
def corners(b):
    return torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), 1).clamp(0, 1)


def giou(p, t):
    lo = torch.maximum(p[:, :2], t[:, :2]); hi = torch.minimum(p[:, 2:], t[:, 2:])
    inter = (hi - lo).clamp(min=0).prod(1)
    ap = (p[:, 2:] - p[:, :2]).clamp(min=0).prod(1); at = (t[:, 2:] - t[:, :2]).clamp(min=0).prod(1)
    union = (ap + at - inter).clamp(min=1e-9)
    enc = (torch.maximum(p[:, 2:], t[:, 2:]) - torch.minimum(p[:, :2], t[:, :2])).clamp(min=0).prod(1).clamp(min=1e-9)
    return inter / union - (enc - union) / enc


def iou(p, t):
    lo = torch.maximum(p[:, :2], t[:, :2]); hi = torch.minimum(p[:, 2:], t[:, 2:])
    inter = (hi - lo).clamp(min=0).prod(1)
    ap = (p[:, 2:] - p[:, :2]).clamp(min=0).prod(1); at = (t[:, 2:] - t[:, :2]).clamp(min=0).prod(1)
    return inter / (ap + at - inter).clamp(min=1e-9)


def objective(out, batch, device, grid):
    pos = batch["present"].to(device) > 0
    labels = batch["label"].to(device); truth = batch["box"].to(device)
    zero = out["class_logits"].sum() * 0
    presence = F.binary_cross_entropy_with_logits(out["presence_logits"], pos.float())
    if not pos.any():
        return presence, {}
    cls = F.cross_entropy(out["class_logits"][pos], labels[pos], label_smoothing=.05)
    b = truth[pos]; target = torch.cat(((b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]), 1)
    pred = out["box_cxcywh"][pos]
    l1 = F.l1_loss(pred, target); g = (1 - giou(corners(pred), b)).mean()
    heat = zero
    if "heat_logits" in out:
        gx = (target[:, 0] * grid).long().clamp(0, grid - 1); gy = (target[:, 1] * grid).long().clamp(0, grid - 1)
        heat = F.cross_entropy(out["heat_logits"][pos], gy * grid + gx)
    loss = cls + presence + 5 * l1 + 2 * g + .5 * heat
    return loss, {"cls": float(cls), "presence": float(presence), "l1": float(l1), "giou": float(g), "heat": float(heat)}


def predict_records(model, dataset, device, batch_size=128):
    records = []; model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            out = model(batch["image"].to(device))
            prob = out["class_logits"].softmax(-1); value, label = prob.max(-1)
            presence = out["presence_logits"].sigmoid(); boxes = corners(out["box_cxcywh"])
            ious = iou(boxes.cpu(), batch["box"])
            for k, index in enumerate(batch["index"].tolist()):
                row = dataset.rows[index]
                records.append({"id": row["id"], "label": row["label"], "present": row["present"], "group": row["group"],
                                "source": row["source"], "predicted_class": int(label[k]), "presence_score": float(presence[k]),
                                "class_score": float(value[k]), "score": float(presence[k] * value[k]),
                                "predicted_box_letterbox": boxes[k].tolist(),
                                "truth_box_letterbox": batch["box"][k].tolist() if row["present"] else None,
                                "iou": float(ious[k]) if row["present"] else None})
    return records


def train(args):
    exp = dict(EXPERIMENTS[args.experiment])
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} exists; use a new folder")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed); device = choose_device(args.device)
    rows = read_jsonl(args.manifest)
    for extra in args.extra_manifest or []:
        rows += read_jsonl(extra)
    counts = validate_rows(rows)
    init = None
    if args.init:  # fine-tune from a previous improved checkpoint (folded, BN-free)
        c = torch.load(args.init, map_location="cpu", weights_only=True)
        if c.get("architecture") != "improved_detector":
            raise ValueError("--init must be an improved detector checkpoint")
        exp = {**c["config"], "transfer": False, "bn": False}
        init = {"checkpoint": str(args.init), "checkpoint_sha256": file_hash(args.init), "epoch": c["epoch"]}
    model = Detector(**exp)
    if args.init:
        model.load_state_dict(c["model_state_dict"], strict=True)
    transfer = transfer_original(model, args.classifier) if exp["transfer"] else None
    model.to(device)
    size = exp["size"]
    t0 = time.perf_counter()
    train_set = DetectionSet(rows, "train", size, training=True, zoom=args.zoom, limit=args.limit)
    val_set = DetectionSet(rows, "val", size, training=False, limit=args.limit)
    rehearsal = RehearsalSet(size, limit=args.limit) if args.rehearsal else None
    load_seconds = time.perf_counter() - t0
    freq = Counter((r["label"], r["source"]) for r in train_set.rows); per_label = Counter(l for l, _ in freq)
    weights = [1 / (per_label[r["label"]] * freq[(r["label"], r["source"])]) for r in train_set.rows]
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=True,
                        sampler=WeightedRandomSampler(weights, len(weights), replacement=True, generator=gen))
    aux = None
    if rehearsal is not None:
        af = Counter(rehearsal.labels)
        aux = DataLoader(rehearsal, batch_size=args.batch_size, num_workers=0, drop_last=True,
                         sampler=WeightedRandomSampler([1 / af[l] for l in rehearsal.labels], len(rehearsal), replacement=True, generator=gen))
    head_params = [p for n, p in model.named_parameters() if not n.startswith("features.") and not n.startswith("fc.")]
    body_params = [p for n, p in model.named_parameters() if n.startswith("features.") or n.startswith("fc.")]
    body_lr = args.lr * (args.transfer_lr_scale if exp["transfer"] else 1.)
    opt = torch.optim.AdamW([{"params": body_params, "lr": body_lr}, {"params": head_params, "lr": args.lr}], weight_decay=1e-4)
    steps = args.epochs * len(loader); warm = 3 * len(loader)
    base = [g["lr"] for g in opt.param_groups]
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    plain = Detector(**{**exp, "bn": False})
    config = {"experiment": args.experiment, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in exp.items()},
              "args": vars(args), "counts": counts, "transfer": transfer, "init": init,
              "extra_manifests": {str(m): file_hash(m) for m in (args.extra_manifest or [])},
              "selection_source": args.select_source,
              "inference_parameters": parameter_count(plain), "macs": macs(plain),
              "manifest_sha256": file_hash(args.manifest), "environment": environment(device),
              "selection": "Validation mean AP50 over present classes, tie-break all-positive mean IoU. Test never used.",
              "load_seconds": load_seconds}
    save_json(output / "config.json", config)
    print(json.dumps({k: config[k] for k in ("experiment", "inference_parameters", "macs", "size", "head")}), flush=True)
    best = (-1., -1.); history = []; step = 0; start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; parts_sum = Counter()
        aux_iter = iter(aux) if aux is not None else None
        for batch in loader:
            for g, b0 in zip(opt.param_groups, base):  # linear warm-up then cosine
                g["lr"] = b0 * (step / warm if step < warm else .5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            out = model(batch["image"].to(device))
            loss, parts = objective(out, batch, device, model.grid)
            if aux_iter is not None:
                try:
                    ax, ay = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(aux); ax, ay = next(aux_iter)
                loss = loss + args.rehearsal_weight * F.cross_entropy(model.classify(ax.to(device)), ay.to(device), label_smoothing=.05)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); step += 1
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            losses.append(float(loss)); parts_sum.update(parts)
        all_records = predict_records(ema, val_set, device)
        records = [r for r in all_records if not args.select_source or args.select_source in r["source"]]
        m = compute(records)
        score = (m["map50_present_classes"] or 0., m["mean_iou_all_positives"] or 0.)
        m_all = compute(all_records)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_map50": score[0], "val_mean_iou": score[1],
               "val_median_iou": m["median_iou_all_positives"], "val_class_accuracy": m["classification_accuracy_all_positives"],
               "allval_map50": m_all["map50_present_classes"], "allval_mean_iou": m_all["mean_iou_all_positives"],
               "allval_class_accuracy": m_all["classification_accuracy_all_positives"],
               "seconds": time.perf_counter() - start, **{k: v / len(loader) for k, v in parts_sum.items()}}
        history.append(row); print(json.dumps(row), flush=True)
        if score > best:
            best = score
            folded = fold_for_inference(ema.cpu()); ema.to(device)
            torch.save({"format_version": 1, "architecture": "improved_detector", "experiment": args.experiment,
                        "config": folded.cfg, "classes": list(CLASSES), "input_size": size,
                        "model_state_dict": {k: v.clone() for k, v in folded.state_dict().items()},
                        "epoch": epoch, "validation": {"map50": score[0], "mean_iou": score[1],
                                                       "class_accuracy": m["classification_accuracy_all_positives"]},
                        "inference_parameters": parameter_count(folded), "macs": config["macs"],
                        "manifest_sha256": config["manifest_sha256"], "quantized": False, "fpga_tested": False,
                        "preprocessing": f"EXIF-correct RGB, aspect-preserving bilinear letterbox {size}, pad128, (pixel-128)/128"},
                       output / "best_detector.pth")
        save_json(output / "history.json", history)
    # Thresholds chosen on validation only, for the selected checkpoint.
    model = load_improved(output / "best_detector.pth", device)
    all_records = predict_records(model, val_set, device)
    records = [r for r in all_records if not args.select_source or args.select_source in r["source"]]
    threshold, presence_threshold = select_thresholds(records)
    val_metrics = compute(records, threshold, presence_threshold)
    by_source = {}
    for src in sorted({r["source"] for r in all_records}):
        sub = [r for r in all_records if r["source"] == src]
        mm = compute(sub, threshold, presence_threshold)
        by_source[src] = {k: mm[k] for k in ("images", "map50_present_classes", "mean_iou_all_positives", "median_iou_all_positives",
                                              "classification_accuracy_all_positives", "accepted_detection_false_positive_rate")}
    summary = {"experiment": args.experiment, "best_epoch": max(history, key=lambda r: (r["val_map50"], r["val_mean_iou"]))["epoch"],
               "epochs": len(history), "training_seconds": time.perf_counter() - start,
               "inference_parameters": config["inference_parameters"], "macs": config["macs"], "input_size": size,
               "score_threshold_selected_on_validation": threshold, "presence_threshold_selected_on_validation": presence_threshold,
               "validation": {k: val_metrics[k] for k in ("map50_present_classes", "map50_95_present_classes", "mean_iou_all_positives",
                                                         "median_iou_all_positives", "classification_accuracy_all_positives",
                                                         "detection_precision", "detection_recall", "accepted_detection_false_positive_rate")},
               "validation_per_class": {k: {kk: v[kk] for kk in ("ap50", "mean_iou_all_positives", "classification_accuracy", "support")}
                                        for k, v in val_metrics["per_class"].items()},
               "validation_by_source": by_source, "selection_source": args.select_source,
               "test_used_for_selection": False, "quantized": False, "fpga_tested": False}
    save_json(output / "summary.json", summary); print("RUN_COMPLETE", json.dumps(summary), flush=True)


def load_improved(path, device):
    c = torch.load(path, map_location="cpu", weights_only=True)
    if c.get("architecture") != "improved_detector":
        raise ValueError("Not an improved detector checkpoint")
    m = Detector(**c["config"], bn=False); m.load_state_dict(c["model_state_dict"], strict=True)
    return m.to(device).eval()


def final_test(args):
    """Evaluate ONE pre-selected run on the held-out test split, once."""
    run = Path(args.final_test).resolve(); out = run / "test_metrics.json"
    if out.exists():
        raise FileExistsError("Test already evaluated for this run; refusing to re-evaluate")
    device = choose_device(args.device); summary = json.loads((run / "summary.json").read_text())
    rows = read_jsonl(args.manifest); model = load_improved(run / "best_detector.pth", device)
    test = DetectionSet(rows, "test", summary["input_size"], training=False)
    records = predict_records(model, test, device)
    m = compute(records, summary["score_threshold_selected_on_validation"], summary["presence_threshold_selected_on_validation"])
    m.update({"split": "test", "run": str(run), "thresholds_from_validation": True})
    save_json(out, m)
    print("TEST_COMPLETE", json.dumps({k: m[k] for k in ("map50_present_classes", "mean_iou_all_positives", "median_iou_all_positives",
                                                         "classification_accuracy_all_positives", "detection_precision", "detection_recall")}), flush=True)


def sweep(args):
    """Run several experiments in parallel subprocesses (one GPU, separate CPU workers)."""
    names = args.sweep.split(","); procs = []
    base = Path(args.output_root)
    for name in names:
        if name not in EXPERIMENTS:
            raise ValueError(name)
        out = base / f"{args.tag}_{name}"; log = open(ROOT / f"improve_{args.tag}_{name}.log", "w")
        cmd = [sys.executable, "-B", "-m", "detection.improve", "--experiment", name, "--output", str(out),
               "--epochs", str(args.epochs), "--manifest", args.manifest]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        procs.append((name, subprocess.Popen(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT), log))
    status = {}
    for name, p, log in procs:
        status[name] = p.wait(); log.close()
    save_json(ROOT / f"improve_{args.tag}_sweep_status.json", status)
    print("SWEEP_COMPLETE", json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(ROOT / "data/training_manifest.jsonl"))
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS)); p.add_argument("--output")
    p.add_argument("--classifier", default=str(PROJECT / "best_model.pth"))
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--transfer-lr-scale", type=float, default=.25)
    p.add_argument("--ema", type=float, default=.998); p.add_argument("--zoom", type=float, default=.5)
    p.add_argument("--rehearsal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rehearsal-weight", type=float, default=.5)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="Smoke test only: use the first N rows per split")
    p.add_argument("--sweep", help="Comma-separated experiment names to run in parallel")
    p.add_argument("--tag", default="improved_v1"); p.add_argument("--output-root", default=str(ROOT / "runs"))
    p.add_argument("--final-test", help="Run folder to evaluate once on the test split")
    p.add_argument("--extra-manifest", action="append", help="Additional reviewed manifest(s), e.g. the owner's webcam frames")
    p.add_argument("--init", help="Fine-tune starting from this improved detector checkpoint")
    p.add_argument("--select-source", help="Choose the checkpoint on validation rows whose source contains this text")
    a = p.parse_args()
    if a.final_test:
        final_test(a)
    elif a.sweep:
        sweep(a)
    else:
        if not a.experiment or not a.output:
            p.error("--experiment and --output are required")
        train(a)


if __name__ == "__main__":
    main()
