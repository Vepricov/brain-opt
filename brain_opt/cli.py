"""JSON-configured library entry point, independent of PlatformAPI."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="brain-opt")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    from .training import TrainingConfig, run_training
    from .simplex_lora import SimplexLoRAConfig
    from .weight_lora import WeightLoRAConfig

    config = payload.pop("config")
    if config.get("simplex") is not None:
        config["simplex"] = SimplexLoRAConfig(**config["simplex"])
    if config.get("weight_lora") is not None:
        config["weight_lora"] = WeightLoRAConfig(**config["weight_lora"])
    result = run_training(config=TrainingConfig(**config), **payload)
    print(json.dumps({key: result[key] for key in ("status", "val_loss", "reload_verified")}))


if __name__ == "__main__":
    main()
