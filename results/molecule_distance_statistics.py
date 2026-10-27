#!/usr/bin/env python3
"""Compute molecular graph-diameter statistics from a saved PyTorch dataset."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, Mol, rdmolops

# Cached molecules may have been serialized with an older RDKit release.
RDLogger.DisableLog("rdApp.warning")


def extract_molecule(sample: Any) -> Mol:
    """Extract the atom-level RDKit molecule from one saved dataset sample."""
    try:
        molecule = sample["features"][1]["nodes"]["atom_nodes"]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            "Sample does not contain features[1]['nodes']['atom_nodes']"
        ) from error

    if not isinstance(molecule, Mol):
        raise TypeError(
            "features[1]['nodes']['atom_nodes'] is not an RDKit molecule: "
            f"{type(molecule).__name__}"
        )
    return molecule


def molecule_diameter(molecule: Mol) -> int:
    """Return the largest finite shortest-path distance in the molecule."""
    if molecule.GetNumAtoms() == 0:
        raise ValueError("Cannot calculate a diameter for a molecule with no atoms")

    distances = rdmolops.GetDistanceMatrix(molecule)
    components = rdmolops.GetMolFrags(molecule, asMols=False)
    return max(
        int(distances[atom_i][atom_j])
        for component in components
        for atom_i in component
        for atom_j in component
    )


def draw_molecule(molecule: Mol, output_path: Path) -> None:
    """Draw an RDKit molecule and save it as an image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(molecule, str(output_path), size=(800, 600))


def iter_samples(dataset: Any) -> Iterable[tuple[int, Any]]:
    """Yield dataset samples while preserving their instance indices."""
    if isinstance(dataset, dict):
        yield from dataset.items()
        return

    try:
        yield from enumerate(dataset)
    except TypeError as error:
        raise TypeError(
            "The loaded .pt file must contain an iterable dataset"
        ) from error


def calculate_statistics(
    dataset: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], Mol]:
    """Calculate per-instance molecular diameters and aggregate statistics."""
    distances: list[dict[str, Any]] = []
    molecules: list[Mol] = []
    for index, sample in iter_samples(dataset):
        molecule = extract_molecule(sample)
        molecules.append(molecule)
        distances.append(
            {
                "index": int(index),
                "num_atoms": molecule.GetNumAtoms(),
                "largest_distance": molecule_diameter(molecule),
                "smiles": Chem.MolToSmiles(molecule),
            }
        )

    if not distances:
        raise ValueError("The loaded dataset contains no samples")

    values = [item["largest_distance"] for item in distances]
    largest_index = values.index(max(values))
    largest_molecule = distances[largest_index]
    largest_rdkit_molecule = molecules[largest_index]
    summary: dict[str, Any] = {
        "num_molecules": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "largest_distance_index": largest_molecule["index"],
        "largest_distance_smiles": largest_molecule["smiles"],
    }
    return distances, summary, largest_rdkit_molecule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the largest topological (bond-path) distance between any "
            "two atoms for every molecule in a saved .pt dataset."
        )
    )
    parser.add_argument("data_path", type=Path, help="Path to the saved data.pt file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for per-molecule distances and aggregate statistics",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=Path("largest_distance_molecule.png"),
        help=(
            "PNG file for the molecule with the largest shortest path "
            "(default: largest_distance_molecule.png)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = torch.load(args.data_path, weights_only=False)
    distances, summary, largest_molecule = calculate_statistics(dataset)

    print(f"Molecules: {summary['num_molecules']}")
    print(f"Largest distance (min): {summary['min']}")
    print(f"Largest distance (max): {summary['max']}")
    print(f"Largest distance (mean): {summary['mean']:.3f}")
    print(f"Largest distance (median): {summary['median']:.3f}")
    print(f"Largest distance (stdev): {summary['stdev']:.3f}")
    print(
        "Molecule with largest shortest path "
        f"(distance={summary['max']}, index={summary['largest_distance_index']}): "
        f"{summary['largest_distance_smiles']}"
    )
    draw_molecule(largest_molecule, args.image_output)
    print(f"Molecule drawing written to {args.image_output}")

    if args.output is not None:
        result = {"summary": summary, "molecules": distances}
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
