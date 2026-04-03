from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import yaml

DATASETS_DIR = Path(__file__).parent / "datasets"


def normalize_dataset_name(name: str) -> str:
    key = name.strip()
    lowered = key.lower()
    aliases = {
        "tmqm": "tmqm",
        "tmQM": "tmqm",
        "pka": "pka",
        "etn": "etn",
        "ionic_conductivity": "conductivity",
        "conductivity": "conductivity",
        "excess": "mixtures",
    }
    return aliases.get(key, aliases.get(lowered, lowered))


def extract_channel_names(raw_channels) -> Optional[List[str]]:
    if raw_channels is None:
        return None
    if isinstance(raw_channels, str):
        return [raw_channels]
    if not isinstance(raw_channels, list):
        return None

    names = []
    for channel in raw_channels:
        if isinstance(channel, dict):
            name = channel.get("name")
        else:
            name = channel
        if name is None:
            return None
        names.append(name)
    return names


def get_original_channel_names(cfg: dict) -> Optional[List[str]]:
    candidates = [
        cfg.get("data", {}).get("init_args", {}).get("target_columns"),
        cfg.get("data", {}).get("init_args", {}).get("target_col"),
        cfg.get("target_columns"),
        cfg.get("target_col"),
        cfg.get("model", {}).get("init_args", {}).get("target_columns"),
        cfg.get("model", {}).get("init_args", {}).get("target_col"),
        cfg.get("channels"),
    ]

    for candidate in candidates:
        names = extract_channel_names(candidate)
        if names:
            return names
    return None


def load_dataset_spec(dataset_name: str) -> dict:
    dataset_key = normalize_dataset_name(dataset_name)
    path = DATASETS_DIR / f"{dataset_key}.yaml"
    if not path.is_file():
        raise ValueError(f"Dataset spec not found for '{dataset_name}' at {path}")
    return yaml.safe_load(path.read_text())


def parse_dataset_from_name(name: str) -> Optional[str]:
    if name.startswith("mist-conductivity"):
        return "conductivity"
    if name.startswith("mist-mixtures") or "excess" in name:
        return "mixtures"

    parts = name.split("-")
    if len(parts) >= 4 and parts[0] == "mist":
        return normalize_dataset_name(parts[3])
    return None


def find_matching_datasets(original_channel_names: List[str]) -> List[str]:
    matches = []
    for path in sorted(DATASETS_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        dataset_channels = {
            channel["name"]
            for channel in spec.get("channels", [])
            if isinstance(channel, dict) and "name" in channel
        }
        if all(name in dataset_channels for name in original_channel_names):
            matches.append(path.stem)
    return matches


def detect_dataset_name(
    cfg: dict,
    *,
    dataset: Optional[str] = None,
    original_channel_names: Optional[List[str]] = None,
    source_names: Sequence[str] = (),
) -> str:
    if dataset is not None:
        dataset_key = normalize_dataset_name(dataset)
        load_dataset_spec(dataset_key)
        return dataset_key

    candidate_names = [
        cfg.get("data", {}).get("name"),
        cfg.get("data", {}).get("init_args", {}).get("name"),
        cfg.get("dataset"),
        cfg.get("model", {}).get("init_args", {}).get("dataset"),
        *(parse_dataset_from_name(name) for name in source_names),
    ]
    candidate_names = [
        normalize_dataset_name(name)
        for name in candidate_names
        if isinstance(name, str) and name.strip()
    ]

    for candidate in candidate_names:
        try:
            spec = load_dataset_spec(candidate)
        except ValueError:
            continue
        dataset_names = [channel["name"] for channel in spec.get("channels", [])]
        if not original_channel_names or all(
            name in dataset_names for name in original_channel_names
        ):
            return candidate

    if original_channel_names:
        matches = find_matching_datasets(original_channel_names)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "Could not uniquely identify dataset: "
                f"channel names match multiple dataset specs {matches}. "
                "Use --dataset to override."
            )

    raise ValueError("Could not identify dataset. Use --dataset to override.")


def resolve_dataset_channels(
    cfg: dict,
    *,
    dataset: Optional[str] = None,
    source_names: Sequence[str] = (),
) -> List[dict]:
    original_channel_names = get_original_channel_names(cfg)
    dataset_name = detect_dataset_name(
        cfg,
        dataset=dataset,
        original_channel_names=original_channel_names,
        source_names=source_names,
    )
    spec = load_dataset_spec(dataset_name)
    dataset_channels = spec.get("channels", [])

    if not original_channel_names:
        return dataset_channels

    by_name = {channel["name"]: channel for channel in dataset_channels}
    missing = [name for name in original_channel_names if name not in by_name]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' does not contain required channels {missing}"
        )

    return [by_name[name] for name in original_channel_names]
