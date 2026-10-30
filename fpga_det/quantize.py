"""PTQ/export of the webcam-fine-tuned box detector (detection/runs/webcam_ft_v1). No retraining.

Integer arithmetic (everything here is what the RTL computes, bit for bit):
  backbone  11 convs, same scheme as fpga.quantize: input q=pixel-128 (scale 1/128), signed INT8
            per-output-channel weights, INT32 bias/MAC, positive <=23-bit multiplier + shift,
            nearest rounding (halfway away from zero), ReLU clamp 0..127.
  tap       conv 4 output, 64x16x16, kept for the box head.
  pooled    (sum over 8x8 + 32) >> 6 of the final 96x8x8 map.
  dot16     INT8 dot product + INT32 bias, requantised to INT16 Q8.8 (scale 1/256), no ReLU.
            class (4), presence (1), size-from-pooled (2): dot16 over pooled.
            heat_coarse: dot16 over final channels per 8x8 cell; heat_fine: per 16x16 tap cell.
  heat      heat_fine[y,x] + heat_coarse[y>>1,x>>1]   (INT32, nearest upsample like the model)
  softmax   e = EXP[min((max-heat+4)>>3, 512)], EXP = round(65535*exp(-i/32)) (same table as classifier)
            S = sum(e). cx16 = (sum(e*(2x+1))*2048 + S//2)//S, likewise cy16 (Q0.16 of the image side)
            att[c] = (sum(e*tap[c]) + S//2)//S, INT8 at the tap scale (soft-argmax attended features)
  size      dot16(pooled, size_p) + dot16(att, size_a, bias 0), saturated to INT16 Q8.8
Host decodes: softmax(class/256), sigmoid(presence/256), box = (cx16/65536, cy16/65536, sigmoid(size/256)).
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from classifier.common import file_hash, seed_everything
from detection.fetch_data import read_jsonl
from detection.improve import DetectionSet, load_improved
from detection.metrics import compute
from fpga.quantize import multiplier, requant, round_away, save_json, write_mem
from . import PROJECT, ROOT

RUN = PROJECT / "detection" / "runs" / "webcam_ft_v1"
MANIFESTS = [PROJECT / "detection" / "data" / "training_manifest.jsonl", PROJECT / "detection" / "data" / "webcam_s1_manifest.jsonl"]
TAP_LAYER = 4
GRID = 16
EXP = round_away(65535 * np.exp(-np.arange(513) / 32)).astype(np.int64)
HEAD_ROWS = ("cls0", "cls1", "cls2", "cls3", "presence", "size_p0", "size_p1", "heat_coarse", "heat_fine", "size_a0", "size_a1")
WEIGHT_DEPTH = 32768
PARAMETER_DEPTH = 1024


def conv_int(x, w, b, m, s, layer):
    k, st, p = layer["kernel"], layer["stride"], layer["pad"]
    padded = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (p, p), (p, p)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(2, 3))[:, :, ::st, ::st]
    n, ci, h, ww, _, _ = windows.shape
    if layer["depthwise"]:
        acc = (windows * w[None, :, None, None, 0, :, :]).sum((-2, -1)) + b[None, :, None, None]
    else:
        patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, ci * k * k)
        acc = (patches @ w.reshape(w.shape[0], -1).T).reshape(n, h, ww, w.shape[0]).transpose(0, 3, 1, 2) + b[None, :, None, None]
    if np.max(np.abs(acc)) >= 2 ** 31:
        raise OverflowError("INT32 convolution overflow")
    return requant(acc, m[None, :, None, None], s[None, :, None, None])


class IntegerDetector:
    def __init__(self, description, arrays):
        self.description = description; self.arrays = {k: np.asarray(v) for k, v in arrays.items()}

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        return cls(json.loads((folder / "model.json").read_text()), dict(np.load(folder / "weights.npz")))

    def dot16(self, x, name):
        a = self.arrays
        acc = x.astype(np.int64) @ a[f"w_{name}"].astype(np.int64).T + a[f"b_{name}"].astype(np.int64)
        if np.max(np.abs(acc)) >= 2 ** 31:
            raise OverflowError(f"INT32 head overflow in {name}")
        return requant(acc, a[f"m_{name}"], a[f"s_{name}"], bits=16, relu=False).astype(np.int64)

    def run(self, q, trace=False):
        x = np.asarray(q, dtype=np.int8)
        if x.ndim == 3:
            x = x[None]
        a = self.arrays; states = []
        for i, layer in enumerate(self.description["layers"]):
            x = conv_int(x, a[f"w{i}"].astype(np.int64), a[f"b{i}"].astype(np.int64), a[f"m{i}"], a[f"s{i}"], layer)
            states.append(x)
        final = states[-1].astype(np.int64); tap = states[TAP_LAYER].astype(np.int64); n = len(x)
        pooled = (final.sum((2, 3)) + 32) >> 6
        cls = self.dot16(pooled, "cls"); presence = self.dot16(pooled, "presence")[:, 0]; size_p = self.dot16(pooled, "size_p")
        coarse = self.dot16(final.reshape(n, 96, 64).transpose(0, 2, 1), "heat_coarse")[..., 0]      # n x 64
        fine = self.dot16(tap.reshape(n, 64, 256).transpose(0, 2, 1), "heat_fine")[..., 0]          # n x 256
        gy, gx = np.divmod(np.arange(GRID * GRID), GRID)
        heat = fine + coarse[:, (gy >> 1) * 8 + (gx >> 1)]
        index = np.clip((heat.max(1, keepdims=True) - heat + 4) >> 3, 0, 512)
        e = EXP[index]; total = e.sum(1)
        cx16 = ((e * (2 * gx + 1)).sum(1) * 2048 + total // 2) // total
        cy16 = ((e * (2 * gy + 1)).sum(1) * 2048 + total // 2) // total
        weighted = np.einsum("ni,nci->nc", e, tap.reshape(n, 64, 256))
        if np.max(weighted) >= 2 ** 32:
            raise OverflowError("attention sum overflow")
        att = (weighted + total[:, None] // 2) // total[:, None]
        size = np.clip(size_p + self.dot16(att, "size_a"), -32768, 32767)
        out = {"class_logits": cls, "presence": presence, "cx16": cx16, "cy16": cy16, "size": size}
        if trace:
            out.update(states=states, pooled=pooled, heat=heat, exp=e, total=total, att=att, size_p=size_p, coarse=coarse, fine=fine)
        return out


def decode(out):
    """Host-side decode of the INT outputs to the float record format used by detection.metrics."""
    logits = out["class_logits"] / 256.0
    prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    presence = 1 / (1 + np.exp(-out["presence"] / 256.0))
    wh = 1 / (1 + np.exp(-out["size"] / 256.0))
    cx = out["cx16"] / 65536.0; cy = out["cy16"] / 65536.0
    boxes = np.clip(np.stack([cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2], 1), 0, 1)
    return prob, presence, boxes


def quantize_weights(weights, input_scale, output_scale, bias=None):
    """Per-output-row symmetric INT8 weights, INT32 bias, requant multiplier/shift."""
    w2 = weights.reshape(weights.shape[0], -1)
    ws = np.maximum(np.abs(w2).max(1) / 127, 1e-12)
    qw = round_away(weights / ws.reshape((-1,) + (1,) * (weights.ndim - 1))).clip(-127, 127).astype(np.int8)
    qb = np.zeros(weights.shape[0], np.int64) if bias is None else round_away(bias / (input_scale * ws)).astype(np.int64)
    if np.max(np.abs(qb)) >= 2 ** 31:
        raise OverflowError("Bias does not fit INT32")
    ms = [multiplier(float(v)) for v in input_scale * ws / output_scale]
    return qw, qb.astype(np.int32), np.array([v[0] for v in ms], np.int32), np.array([v[1] for v in ms], np.int32)


def export(model, ranges, percentile, calibration):
    convs = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    layers = []; arrays = {}; scale = 1 / 128; woff = 0; poff = 0; hw = model.cfg["size"]; scales = []
    for i, conv in enumerate(convs):
        out_scale = max(float(np.quantile(ranges[i], percentile)), 1e-8) / 127
        qw, qb, m, s = quantize_weights(conv.weight.detach().cpu().numpy(), scale, out_scale, conv.bias.detach().cpu().numpy())
        arrays.update({f"w{i}": qw, f"b{i}": qb, f"m{i}": m, f"s{i}": s})
        k, st, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]; out = (hw + 2 * p - k) // st + 1
        layers.append({"cin": conv.in_channels, "cout": conv.out_channels, "kernel": k, "stride": st, "pad": p,
                       "depthwise": conv.groups == conv.in_channels and conv.groups > 1, "input_hw": hw, "output_hw": out,
                       "weight_offset": woff, "parameter_offset": poff, "input_scale": scale, "output_scale": out_scale})
        woff += qw.size; poff += qw.shape[0]; scale = out_scale; hw = out; scales.append(out_scale)
    final_scale, tap_scale = scales[-1], scales[TAP_LAYER]
    d = lambda t: t.detach().cpu().numpy()
    size_w, size_b = d(model.size_fc.weight), d(model.size_fc.bias)
    heads = {"cls": (d(model.fc.weight), d(model.fc.bias), final_scale),
             "presence": (d(model.presence.weight), d(model.presence.bias), final_scale),
             "size_p": (size_w[:, :96], size_b, final_scale),
             "heat_coarse": (d(model.heat_coarse.weight).reshape(1, 96), d(model.heat_coarse.bias), final_scale),
             "heat_fine": (d(model.heat_fine.weight).reshape(1, 64), d(model.heat_fine.bias), tap_scale),
             "size_a": (size_w[:, 96:], None, tap_scale)}
    head_layout = {}
    for name, (w, b, s_in) in heads.items():
        qw, qb, m, s = quantize_weights(w, s_in, 1 / 256, b)
        arrays.update({f"w_{name}": qw, f"b_{name}": qb, f"m_{name}": m, f"s_{name}": s})
        head_layout[name] = {"rows": int(qw.shape[0]), "inputs": int(qw.shape[1]), "weight_offset": woff, "parameter_offset": poff}
        woff += qw.size; poff += qw.shape[0]
    if woff > WEIGHT_DEPTH or poff > PARAMETER_DEPTH:
        raise ValueError("Export does not fit the weight/parameter ROMs")
    desc = {"architecture": "improved_detector_w128_sa_int8", "classes": ["rubiks_cube", "pen", "book", "mouse"],
            "input": [3, 128, 128], "checkpoint_sha256": file_hash(RUN / "best_detector.pth"), "layers": layers,
            "tap_layer": TAP_LAYER, "heads": head_layout, "weights_used": woff, "parameters_used": poff,
            "activation_percentile": percentile, "input_scale": 1 / 128, "output_scale_q8_8": 1 / 256,
            "calibration_image_ids": calibration, "retrained": False}
    return IntegerDetector(desc, arrays)


def records_for(dataset, outputs):
    prob, presence, boxes = outputs
    records = []
    for k, row in enumerate(dataset.rows):
        cls = int(prob[k].argmax()); box = boxes[k]; truth = dataset.items[k][1]
        iou = None
        if row["present"]:
            lo = np.maximum(box[:2], truth[:2]); hi = np.minimum(box[2:], truth[2:]); inter = np.prod(np.clip(hi - lo, 0, None))
            union = np.prod(np.clip(box[2:] - box[:2], 0, None)) + np.prod(np.clip(truth[2:] - truth[:2], 0, None)) - inter
            iou = float(inter / max(union, 1e-9))
        records.append({"id": row["id"], "label": row["label"], "present": row["present"], "group": row["group"], "source": row["source"],
                        "predicted_class": cls, "presence_score": float(presence[k]), "class_score": float(prob[k, cls]),
                        "score": float(presence[k] * prob[k, cls]), "iou": iou})
    return records


def summarize(records, threshold, presence_threshold):
    keys = ("map50_present_classes", "mean_iou_all_positives", "median_iou_all_positives", "classification_accuracy_all_positives",
            "detection_precision", "detection_recall", "accepted_detection_false_positive_rate")
    m = compute(records, threshold, presence_threshold)
    out = {k: m[k] for k in keys}
    out["per_class"] = {k: {kk: v[kk] for kk in ("ap50", "mean_iou_all_positives", "classification_accuracy", "support")}
                        for k, v in m["per_class"].items()}
    return out


def float_outputs(model, images):
    with torch.inference_mode():
        out = model(torch.from_numpy(images))
    prob = out["class_logits"].softmax(-1).numpy(); presence = out["presence_logits"].sigmoid().numpy()
    b = out["box_cxcywh"].numpy()
    boxes = np.clip(np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1), 0, 1)
    return prob, presence, boxes


def run_integer(reference, q, batch=32):
    parts = [reference.run(q[i:i + batch]) for i in range(0, len(q), batch)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def write_export(reference, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "model.json", reference.description); np.savez(folder / "weights.npz", **reference.arrays)
    a = reference.arrays; weights = []; meta = []
    names = [str(i) for i in range(len(reference.description["layers"]))] + ["_" + n for n in reference.description["heads"]]
    for suffix in names:
        weights.extend(int(v) for v in a["w" + suffix].reshape(-1))
        for b, m, s in zip(a["b" + suffix], a["m" + suffix], a["s" + suffix]):
            meta.append(((int(b) & 0xffffffff) << 32) | (int(m) << 8) | int(s))
    weights.extend([0] * (WEIGHT_DEPTH - len(weights))); meta.extend([0] * (PARAMETER_DEPTH - len(meta)))
    write_mem(folder / "weights.mem", weights, 8); write_mem(folder / "parameters.mem", meta, 64)
    write_mem(folder / "exp.mem", EXP, 16)
    model_id = hashlib.sha256(b"".join((folder / n).read_bytes() for n in ("weights.mem", "parameters.mem", "exp.mem"))).hexdigest()[:16]
    (folder / "hardware_id.svh").write_text(f"localparam [63:0] MODEL_ID=64'h{model_id};\n", encoding="ascii")
    save_json(folder / "hardware_identity.json", {"model_id_hex": model_id, "source_checkpoint_sha256": reference.description["checkpoint_sha256"]})
    layers = reference.description["layers"]; n = len(layers)
    # Activation memory map (two 64 KiB banks, A=0/B=1). Source bank alternates A,B,A,...; offsets keep
    # the tap (conv 4 output, bank B @0) intact while layers 5..10 use A@0 and B@32768.
    src_off = [0] * n; dst_off = [0] * n
    for i in range(n):
        if i > TAP_LAYER and i % 2 == 0:
            dst_off[i] = 32768
        if i > TAP_LAYER + 1 and i % 2 == 1:
            src_off[i] = 32768
    for i, l in enumerate(layers):
        src_bank, dst_bank = i % 2, 1 - i % 2
        if l["input_hw"] ** 2 * l["cin"] + src_off[i] > 65536 or l["output_hw"] ** 2 * l["cout"] + dst_off[i] > 65536:
            raise ValueError("activation map exceeds a 64 KiB bank")
        if dst_bank == 1 and i > TAP_LAYER and dst_off[i] < 64 * 256:
            raise ValueError("layer would overwrite the tap")
    lines = ["// Generated by fpga_det.quantize from detection/runs/webcam_ft_v1. Do not hand-edit.",
             f"localparam integer NUM_LAYERS = {n};"]
    table = {"CIN": [l["cin"] for l in layers], "COUT": [l["cout"] for l in layers], "KERNEL": [l["kernel"] for l in layers],
             "STRIDE": [l["stride"] for l in layers], "PAD": [l["pad"] for l in layers], "INPUT_HW": [l["input_hw"] for l in layers],
             "OUTPUT_HW": [l["output_hw"] for l in layers], "WEIGHT_OFFSET": [l["weight_offset"] for l in layers],
             "PARAMETER_OFFSET": [l["parameter_offset"] for l in layers], "INPUT_PLANE": [l["input_hw"] ** 2 for l in layers],
             "OUTPUT_PLANE": [l["output_hw"] ** 2 for l in layers], "KERNEL_AREA": [l["kernel"] ** 2 for l in layers],
             "WEIGHT_STEP": [(1 if l["depthwise"] else l["cin"]) * l["kernel"] ** 2 for l in layers],
             "SRC_OFFSET": src_off, "DST_OFFSET": dst_off}
    for key, vals in table.items():
        lines.append(f"localparam integer L_{key} [0:{n - 1}] = '{{{','.join(str(int(v)) for v in vals)}}};")
    lines.append(f"localparam bit L_DEPTHWISE [0:{n - 1}] = '{{" + ",".join("1'b1" if l["depthwise"] else "1'b0" for l in layers) + "};")
    for name, h in reference.description["heads"].items():
        lines.append(f"localparam integer H_{name.upper()}_W = {h['weight_offset']};")
        lines.append(f"localparam integer H_{name.upper()}_P = {h['parameter_offset']};")
    (folder / "layers.svh").write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(ROOT / "export"))
    p.add_argument("--calibration", type=int, default=400)
    args = p.parse_args()
    output = Path(args.output)
    if (output / "model.json").exists():
        raise FileExistsError("Existing integer export; choose another output directory")
    seed_everything(42)
    summary = json.loads((RUN / "summary.json").read_text())
    threshold, presence_threshold = summary["score_threshold_selected_on_validation"], summary["presence_threshold_selected_on_validation"]
    model = load_improved(RUN / "best_detector.pth", "cpu")
    rows = []
    for m in MANIFESTS:
        rows += read_jsonl(m)
    # Calibration: TRAIN split only, balanced over class/negatives and source.
    train = [r for r in rows if r.get("split") == "train"]
    groups = defaultdict(list)
    for r in train:
        groups[(r["label"] if r["present"] else -1, "webcam" in r["source"])].append(r)
    rng = random.Random(42); calib = []
    for key in sorted(groups, key=str):
        calib += rng.sample(groups[key], min(len(groups[key]), args.calibration // len(groups)))
    started = time.perf_counter()
    cal_set = DetectionSet(calib, "train", 128, training=False)
    cal_images = np.stack([np.asarray(img, np.float32).transpose(2, 0, 1) for img, _ in cal_set.items]) / 128 - 1
    convs = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
    ranges = defaultdict(list); hooks = []
    for i, conv in enumerate(convs):
        hooks.append(conv.register_forward_hook(lambda m, inp, out, i=i: ranges[i].append(out.detach().relu().numpy().reshape(-1))))
    with torch.inference_mode():
        for s in range(0, len(cal_images), 64):
            model(torch.from_numpy(cal_images[s:s + 64].astype(np.float32)))
    for h in hooks:
        h.remove()
    ranges = {k: np.concatenate(v) for k, v in ranges.items()}
    print(f"calibration images {len(calib)} ({time.perf_counter() - started:.1f}s)", flush=True)

    val = DetectionSet(rows, "val", 128, training=False)
    images = np.stack([np.asarray(img, np.float32).transpose(2, 0, 1) for img, _ in val.items])
    q = (images - 128).astype(np.int8); fimages = (images - 128) / 128
    fp_records = records_for(val, float_outputs(model, fimages.astype(np.float32)))
    webcam = lambda recs: [r for r in recs if "webcam" in r["source"]]
    result = {"fp32": {"webcam_val": summarize(webcam(fp_records), threshold, presence_threshold),
                       "all_val": summarize(fp_records, threshold, presence_threshold)}, "candidates": []}
    print("FP32 webcam val", json.dumps({k: v for k, v in result["fp32"]["webcam_val"].items() if k != "per_class"}), flush=True)
    best = None
    for percentile in (.999, .9999, 1.):
        ref = export(model, ranges, percentile, [r["id"] for r in calib])
        recs = records_for(val, decode(run_integer(ref, q)))
        wm = summarize(webcam(recs), threshold, presence_threshold); am = summarize(recs, threshold, presence_threshold)
        result["candidates"].append({"percentile": percentile, "webcam_val": wm, "all_val": am})
        print("INT8", percentile, json.dumps({k: v for k, v in wm.items() if k != "per_class"}), flush=True)
        key = (wm["map50_present_classes"] or 0, wm["mean_iou_all_positives"] or 0)
        if best is None or key > best[0]:
            best = (key, ref, percentile)
    ref = best[1]; write_export(ref, output)
    result.update({"selected_percentile": best[2], "selection": "webcam validation mAP50, tie-break mean IoU (same rule as training)",
                   "thresholds": {"score": threshold, "presence": presence_threshold}, "test_used": False,
                   "calibration_images": len(calib), "calibration_split": "train only"})
    save_json(output / "quantization_validation.json", result)
    # Predetermined simulation vectors: first webcam validation frame of each class plus the first negative.
    vectors = output / "vectors"; vectors.mkdir(exist_ok=True)
    picks = []
    for want in (0, 1, 2, 3, -1):
        for k, r in enumerate(val.rows):
            if "webcam" in r["source"] and (r["label"] if r["present"] else -1) == want:
                picks.append(k); break
    cases = []
    for c, k in enumerate(picks):
        out = ref.run(q[k], trace=True)
        write_mem(vectors / f"input{c}.mem", q[k], 8)
        for layer, state in enumerate(out["states"]):
            write_mem(vectors / f"case{c}_layer{layer}.mem", state[0], 8)
        expected = [*out["class_logits"][0], out["presence"][0], out["cx16"][0], out["cy16"][0], *out["size"][0]]
        write_mem(vectors / f"case{c}_pooled.mem", out["pooled"][0], 8)
        write_mem(vectors / f"case{c}_att.mem", out["att"][0], 8)
        write_mem(vectors / f"case{c}_heat.mem", out["heat"][0], 32)
        write_mem(vectors / f"case{c}_result.mem", expected, 16)
        cases.append({"case": c, "id": val.rows[k]["id"], "label": val.rows[k]["label"], "present": val.rows[k]["present"],
                      "path": val.rows[k]["path"], "result": [int(v) for v in expected]})
    save_json(vectors / "cases.json", cases)
    print("Exported", output, "percentile", best[2], flush=True)


if __name__ == "__main__":
    main()
