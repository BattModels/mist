#!/usr/bin/env python3
"""
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "jinja2>=3.0.0",
#     "pyyaml>=6.0",
# ]
# ///
"""

import argparse
import re
import sys
from pathlib import Path
import json
import yaml
from jinja2 import Environment, FileSystemLoader, Template


def parse_model_dir_name(model_dir_name: str) -> tuple[str, str]:
    """
    Parse model directory name to extract encoder and dataset.

    Expected formats:
        mist-28M-{id}-{dataset}
        mist-1.8B-{id}-{dataset}
        mist-26.9M-{id}-{dataset}
        mist-conductivity-{size}-{id}
        mist-mixtures-{id}
    """
    # Handle special cases
    if model_dir_name.startswith("mist-conductivity"):
        # mist-conductivity-27.0M-2mpg8dcd -> (mist-28M, ionic_conductivity)
        match = re.match(r"mist-conductivity-(\d+\.?\d*)M?-", model_dir_name)
        if match:
            size = float(match.group(1))
            encoder = "mist-28M" if size < 100 else "mist-1.8B"
            return encoder, "ionic_conductivity"

    if model_dir_name.startswith("mist-mixtures"):
        # mist-mixtures-zffffbex -> (mist-28M, mixtures)
        return "mist-28M", "mixtures"

    # Standard format: mist-{size}-{id}-{dataset}
    parts = model_dir_name.split("-")
    if len(parts) >= 4:
        encoder_size = parts[1]  # e.g., "28M", "1.8B", "26.9M"
        dataset = parts[3]  # e.g., "qm9", "freesolv"

        # Normalize encoder size
        if encoder_size in ["26.9M", "27.0M", "28M"]:
            encoder_key = "mist-28M"
        elif encoder_size == "1.8B":
            encoder_key = "mist-1.8B"
        else:
            encoder_key = f"mist-{encoder_size}"

        # Handle dataset variations
        dataset_key = dataset.lower()

        # Map variations
        dataset_map = {
            "tmqm": "tmQM",
        }

        dataset_key = dataset_map.get(dataset_key, dataset_key)

        return encoder_key, dataset_key

    raise ValueError(f"Could not parse model directory name: {model_dir_name}")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_model_card(
    encoder_key: str,
    dataset_key: str,
    config: dict,
    template: Template,
) -> str:
    encoder = config["encoders"][encoder_key]
    dataset = config["datasets"][dataset_key]

    return template.render(encoder=encoder, dataset=dataset)


def generate_model_card_for_directory(
    model_dir: Path,
    ckpt_path: Path | None = None,
) -> str:
    config_path = Path(__file__).parent / "model_configs.yaml"
    template_path = Path(__file__).parent / "model_card_template.j2"

    try:
        encoder_key, dataset_key = parse_model_dir_name(model_dir.name)
    except ValueError:
        model_config = None

        # First try checkpoint path if provided
        if ckpt_path is not None:
            ckpt_config_path = ckpt_path.parent.parent / "config.yaml"
            if ckpt_config_path.exists():
                import yaml

                with open(ckpt_config_path, "r") as f:
                    model_config = yaml.safe_load(f)

        # Fall back to model directory config.json
        if model_config is None:
            model_config_path = model_dir / "config.json"
            if model_config_path.exists():
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)

        if model_config is None:
            raise ValueError(
                f"Could not parse model directory name '{model_dir.name}' "
                "and no config found in checkpoint or model directory"
            )

        dataset_key = model_config.get("data", {}).get("name")
        if not dataset_key:
            raise ValueError(
                f"Could not parse model directory name '{model_dir.name}' "
                "and config does not contain data.name"
            )

        parts = model_dir.name.split("-")
        if len(parts) >= 2:
            encoder_size = parts[1]
            if encoder_size in ["26.9M", "27.0M", "28M"]:
                encoder_key = "mist-28M"
            elif encoder_size == "1.8B":
                encoder_key = "mist-1.8B"
            else:
                encoder_key = f"mist-{encoder_size}"
        else:
            raise ValueError(
                f"Could not determine encoder from directory name: {model_dir.name}"
            )

    config = load_config(config_path)
    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)

    # Generate and return card
    return generate_model_card(encoder_key, dataset_key, config, template)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIST model cards from Jinja template and YAML config"
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Encoder key (e.g., mist-28M, mist-1.8B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset key (e.g., qm9, freesolv, tox21)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Path to model directory (will auto-parse encoder and dataset)",
    )
    # Config and template paths
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "model_configs.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).parent / "model_card_template.j2",
        help="Path to Jinja template file",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: README.md in model directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated card instead of writing to file",
    )

    args = parser.parse_args()

    # Load config and template
    config = load_config(args.config)
    env = Environment(loader=FileSystemLoader(args.template.parent))
    template = env.get_template(args.template.name)

    # Determine encoder, dataset, and output path
    if args.model_dir:
        # Auto-parse from directory name
        encoder_key, dataset_key = parse_model_dir_name(args.model_dir.name)
        output_path = args.output or (args.model_dir / "README.md")
    elif args.encoder and args.dataset:
        # Explicit encoder and dataset
        encoder_key = args.encoder
        dataset_key = args.dataset
        output_path = args.output or Path("README.md")
    else:
        parser.error("Must specify either --model-dir OR both --encoder and --dataset")

    # Generate card
    model_card = generate_model_card(encoder_key, dataset_key, config, template)

    if args.dry_run:
        print(model_card)
    else:
        output_path.write_text(model_card)


if __name__ == "__main__":
    main()
