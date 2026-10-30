"""No pretrained networks, batch normalization, or hidden expansion layers."""
import copy
import json
import torch
from torch import nn

from . import INPUT_SIZE


class TinyClassifier(nn.Module):
    def __init__(self, architecture="baseline"):
        super().__init__()
        if architecture not in ("baseline", "context", "context2", "standard"):
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        layers = [nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=True), nn.ReLU()]
        previous = 8
        for channels in (16, 32, 48):
            if architecture == "standard":
                layers += [nn.Conv2d(previous, channels, 3, 2, 1), nn.ReLU()]
            else:
                layers += [
                    nn.Conv2d(previous, previous, 3, 2, 1, groups=previous), nn.ReLU(),
                    nn.Conv2d(previous, channels, 1), nn.ReLU(),
                ]
            previous = channels
        for _ in range({"context": 1, "context2": 2}.get(architecture, 0)):
            layers += [nn.Conv2d(48, 48, 3, padding=1, groups=48), nn.ReLU(),
                       nn.Conv2d(48, 48, 1), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(48, 4)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def training_model(architecture="baseline", batch_norm=False):
    model = TinyClassifier(architecture)
    if batch_norm:
        layers = []
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                # Bias is redundant before BN. Folding restores inference biases.
                layer.register_parameter("bias", None)
                layers.extend((layer, nn.BatchNorm2d(layer.out_channels)))
            else:
                layers.append(layer)
        model.features = nn.Sequential(*layers)
    return model


def export_inference_model(model):
    """Fold optional training BN into ordinary Conv2d layers on CPU."""
    result = copy.deepcopy(model).cpu().eval()
    original, layers, index = list(result.features), [], 0
    while index < len(original):
        layer = original[index]
        if (isinstance(layer, nn.Conv2d) and index + 1 < len(original)
                and isinstance(original[index + 1], nn.BatchNorm2d)):
            layer = torch.nn.utils.fusion.fuse_conv_bn_eval(layer, original[index + 1])
            index += 1
        layers.append(layer)
        index += 1
    result.features = nn.Sequential(*layers)
    return result


def describe_model(architecture="baseline"):
    model = TinyClassifier(architecture)
    rows, hooks = [], []

    def record(name):
        def hook(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                terms = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
                macs = output.numel() * terms
            else:
                macs = module.in_features * module.out_features
            rows.append({"layer": name, "shape": list(output.shape[1:]),
                         "weights": module.weight.numel(), "biases": module.bias.numel(),
                         "macs": macs, "activation_int8_bytes": output.numel()})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(record(name)))
    with torch.inference_mode():
        model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
    for hook in hooks:
        hook.remove()
    weights = sum(row["weights"] for row in rows)
    biases = sum(row["biases"] for row in rows)
    return {"architecture": architecture, "input": [3, INPUT_SIZE, INPUT_SIZE],
            "parameters": weights + biases, "weights": weights, "biases": biases,
            "fp32_parameter_bytes": 4 * (weights + biases),
            "future_int8_weight_bytes": weights, "future_int32_bias_bytes": 4 * biases,
            "macs": sum(row["macs"] for row in rows),
            "future_peak_live_activation_bytes": 46080,
            "note": "INT8 figures are analytical payload estimates, not a quantized checkpoint or synthesis result.",
            "layers": rows}


if __name__ == "__main__":
    descriptions = [describe_model(name) for name in ("baseline", "context", "context2", "standard")]
    assert descriptions[0]["parameters"] == 3252
    assert descriptions[0]["macs"] == 773184
    print(json.dumps(descriptions, indent=2))

