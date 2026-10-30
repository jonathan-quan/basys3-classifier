import argparse
import json

import torch

from .common import choose_device, environment, seed_everything, write_json
from .model import TinyClassifier, describe_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="environment.json")
    args = parser.parse_args()
    seed_everything(42)
    device = choose_device()
    model = TinyClassifier().to(device)
    inputs = torch.randn(4, 3, 96, 96, device=device)
    model(inputs).square().mean().backward()
    report = {**environment(device), "forward_backward_check": "passed",
              "model": describe_model(), "cuda_architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else []}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

