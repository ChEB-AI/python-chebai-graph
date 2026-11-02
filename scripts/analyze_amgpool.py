#!/usr/bin/env python3
"""Measure representation scales and classifier contributions in AMGPool models.

Example (preferred; uses hyperparameters stored in the checkpoint):
    python scripts/analyze_amgpool.py \
        --checkpoint path/to/model.ckpt \
        --output-dir analysis/bbbp

Explicit model and data configs can also be supplied together when the
checkpoint does not contain Lightning hyperparameters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from chebai.models.base import ChebaiBaseNet
from chebai.preprocessing.datasets.base import XYBaseDataModule
from lightning.pytorch.cli import instantiate_module

LEVELS = ("atom", "motif", "graph")


def _import_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _instantiate(config: dict[str, Any]) -> Any:
    class_path = config.get("class_path")
    if not class_path:
        raise ValueError("Expected a config with a 'class_path' entry.")
    init_args = config.get("init_args", {})
    return _import_class(class_path)(**init_args)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} does not contain a state dictionary.")

    model_state = model.state_dict()
    candidates = []
    for prefix in ("", "model.", "module."):
        candidates.append(
            {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        )
    matching = max(
        (candidate for candidate in candidates if candidate),
        key=lambda candidate: len(set(candidate) & set(model_state)),
        default={},
    )
    filtered = {key: value for key, value in matching.items() if key in model_state}
    if not filtered:
        raise ValueError(f"{path} contains no weights matching the configured model.")
    result = model.load_state_dict(filtered, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Checkpoint {path} is missing model weights: "
            + ", ".join(sorted(result.missing_keys))
        )


def _load_experiment(
    checkpoint_path: Path,
) -> tuple[ChebaiBaseNet, XYBaseDataModule]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        data_hparams = copy.deepcopy(checkpoint["datamodule_hyper_parameters"])
        model_hparams = copy.deepcopy(checkpoint["hyper_parameters"])
    except KeyError as error:
        raise ValueError(
            "Checkpoint must contain Lightning model and datamodule hyperparameters "
            "when --model-config and --data-config are omitted."
        ) from error

    data_hparams.pop("_instantiator", None)
    model_hparams.pop("_instantiator", None)
    data = instantiate_module(XYBaseDataModule, data_hparams)

    load_kwargs = (
        {"pretrained_checkpoint": None}
        if "pretrained_checkpoint" in model_hparams
        else {}
    )
    model = ChebaiBaseNet.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        **load_kwargs,
    )
    return model, data


def _prepare_model_batch(model: ChebaiBaseNet, batch: Any, batch_index: int) -> dict:
    """Apply the repository's standard XYData-to-model batch conversion."""
    if not hasattr(batch, "to") or not hasattr(batch, "x"):
        raise TypeError(
            "The test dataloader must return an XYData batch with an x attribute."
        )
    return model._process_batch(batch.to(model.device), batch_index)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(summary: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = {
        "embedding_rms": [summary["embedding"][level]["rms"] for level in LEVELS],
        "weight_fraction": [summary["classifier"][level]["alpha"] for level in LEVELS],
        "logit_fraction": [summary["contribution"][level]["beta"] for level in LEVELS],
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, values) in zip(axes, metrics.items()):
        axis.bar(LEVELS, values)
        axis.set_title(title.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "representation_summary.png", dpi=180)
    plt.close(fig)


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    model, data = _load_experiment(
        Path(args.checkpoint),
    )
    model.to(args.device).eval()

    linear_layers = [
        layer for layer in model.lin_sequential if isinstance(layer, torch.nn.Linear)
    ]
    if len(linear_layers) != 1:
        raise ValueError(
            "This analysis requires a single-layer linear classifier; "
            f"found {len(linear_layers)} linear layers."
        )
    classifier = linear_layers[0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = data.test_dataloader()
    if isinstance(loader, (list, tuple)):
        if len(loader) != 1:
            raise ValueError("Multiple dataloaders are not supported by this script.")
        loader = loader[0]

    sums = {level: {"std": 0.0, "rms": 0.0, "l2": 0.0} for level in LEVELS}
    contribution_sum = {level: 0.0 for level in LEVELS}
    class_contribution = torch.zeros((classifier.out_features, 3), dtype=torch.float64)
    ablation_sum = torch.zeros(3, dtype=torch.float64)
    sample_count = 0
    batch_limit = args.max_batches

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            batch = _prepare_model_batch(model, raw_batch, batch_index)
            representations = model.pool_representations(batch)
            batch_size = representations[0].shape[0]
            sample_count += batch_size
            blocks = []
            for index, (level, representation) in enumerate(
                zip(LEVELS, representations)
            ):
                representation_cpu = representation.detach().cpu()
                sums[level]["std"] += (
                    float(representation_cpu.std().item()) * batch_size
                )
                sums[level]["rms"] += (
                    float(representation_cpu.square().mean().sqrt().item()) * batch_size
                )
                sums[level]["l2"] += (
                    float(representation_cpu.norm(dim=1).mean().item()) * batch_size
                )
                start = index * representation.shape[1]
                weight = classifier.weight[:, start : start + representation.shape[1]]
                contribution = representation @ weight.t()
                contribution_sum[level] += (
                    float(contribution.norm(dim=1).mean().item()) * batch_size
                )
                class_contribution[:, index] += (
                    contribution.abs().sum(dim=0).detach().cpu().double()
                )
                blocks.append(representation)
            full_logits = classifier(torch.cat(blocks, dim=1))
            for index in range(3):
                ablated = list(blocks)
                ablated[index] = torch.zeros_like(ablated[index])
                ablation_sum[index] += (
                    (full_logits - classifier(torch.cat(ablated, dim=1)))
                    .abs()
                    .sum()
                    .detach()
                    .cpu()
                    .double()
                )

    if sample_count == 0:
        raise RuntimeError("The selected split produced no samples.")
    widths = [classifier.in_features // len(LEVELS)] * len(LEVELS)
    if classifier.in_features % len(LEVELS):
        raise ValueError(
            "The classifier input dimension is not divisible into three AMGPool blocks."
        )
    weight_norms = []
    for index, level in enumerate(LEVELS):
        width = widths[index]
        weight_norms.append(
            float(
                classifier.weight[:, index * width : (index + 1) * width].norm().item()
            )
        )
    weight_total = sum(weight_norms)
    contribution_total = sum(contribution_sum.values())
    if weight_total == 0 or contribution_total == 0:
        raise RuntimeError(
            "The checkpoint has zero total classifier weight or logit contribution; "
            "normalized shares are undefined."
        )
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": "test",
        "samples": sample_count,
        "embedding": {
            level: {
                metric: value / sample_count for metric, value in sums[level].items()
            }
            for level in LEVELS
        },
        "classifier": {
            level: {
                "frobenius_norm": weight_norms[index],
                "alpha": weight_norms[index] / weight_total,
            }
            for index, level in enumerate(LEVELS)
        },
        "contribution": {
            level: {
                "mean_logit_l2": contribution_sum[level] / sample_count,
                "beta": contribution_sum[level] / contribution_total,
            }
            for level in LEVELS
        },
        "ablation_logit_change_mean": {
            level: float(
                ablation_sum[index].item() / sample_count / classifier.out_features
            )
            for index, level in enumerate(LEVELS)
        },
    }
    class_rows = [
        {
            "class_index": index,
            **{
                f"{level}_mean_abs_logit_contribution": float(
                    class_contribution[index, level_index].item() / sample_count
                )
                for level_index, level in enumerate(LEVELS)
            },
        }
        for index in range(classifier.out_features)
    ]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(out_dir / "class_contributions.csv", class_rows)
    _plot_summary(summary, out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("amgpool_analysis"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    analyse(parser.parse_args())


if __name__ == "__main__":
    main()
