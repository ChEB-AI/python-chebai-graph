# This code file is taken from https://github.com/thaonguyen217/farm_molecular_representation
# Reference Paper: Nguyen, Thao, et al. "FARM: Functional Group-Aware Representations for Small Molecules."
# arXiv preprint arXiv:2410.02082 (2024).

import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolToSmiles as m2s

from .fg_constants import ELEMENTS, FLAG_NO_FG


def ring_size_processing(ring_size):
    if ring_size[0] > ring_size[-1]:
        return list(reversed(ring_size))
    else:
        return ring_size


# Function to find all rings connected to a given ring
def find_connected_rings(ring, remaining_rings) -> list[set[int]]:
    connected_rings = [ring]
    merged = True
    while merged:
        merged = False
        for other_ring in remaining_rings:
            if ring & other_ring:  # If there is a shared atom, they are connected
                connected_rings.append(other_ring)
                remaining_rings.remove(other_ring)
                ring = ring.union(other_ring)
                merged = True
    return connected_rings


def set_ring_properties(mol: Chem.Mol) -> list[list[set[int]]] | None:
    if mol is None:
        return

    for atom in mol.GetAtoms():
        atom.SetProp("RING", "")

    AllChem.GetSymmSSSR(mol)

    ######## SET RING PROP ########
    # Get ring information
    ring_info = mol.GetRingInfo()
    fused_rings_groups: list[list[set[int]]] = []

    if ring_info.NumRings() > 0:
        # Get list of atom rings
        atom_rings = ring_info.AtomRings()

        # Initialize a list to hold fused ring blocks and their sizes
        fused_ring_blocks = []
        ring_sizes = []

        # Set of rings to process
        remaining_rings = [set(ring) for ring in atom_rings]

        # Process each ring block
        while remaining_rings:
            ring = remaining_rings.pop(0)
            connected_rings: list[set[int]] = find_connected_rings(
                ring, remaining_rings
            )

            if len(connected_rings) > 1:
                fused_rings_groups.append(connected_rings)

            for ring in connected_rings:
                ring_size = len(ring)
                for atom_idx in ring:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if not atom.HasProp("RING"):
                        atom.SetProp("RING", f"{ring_size}")
                    else:
                        # An atom shared across multiple rings in fused-ring will have Ring size like "5-6-6"
                        atom.SetProp("RING", f"{atom.GetProp('RING')}-{ring_size}")

            # Merge all connected rings into one fused block
            fused_block = set().union(*connected_rings)
            fused_ring_blocks.append(sorted(fused_block))
            ring_sizes.append([len(r) for r in connected_rings])

        # Display the fused ring blocks and their ring sizes
        for i, block in enumerate(fused_ring_blocks):
            rs = "-".join(str(size) for size in ring_size_processing(ring_sizes[i]))
            for idx in block:
                atom = mol.GetAtomWithIdx(idx)
                atom.SetProp("RING", rs)

    return fused_rings_groups


def detect_functional_group(mol: Chem.Mol):
    if mol is None:
        return

    for atom in mol.GetAtoms():
        atom.SetProp("FG", "")

    ######## SET FUNCTIONAL GROUP PROP ########
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_neighbors = atom.GetNeighbors()
        atom_num_neighbors = len(atom_neighbors)
        num_H = atom.GetTotalNumHs()
        in_ring = atom.IsInRing()
        atom_idx = atom.GetIdx()
        charge = atom.GetFormalCharge()

        ########################### Groups containing oxygen ###########################
        if atom_symbol in ["C", "*"] and charge == 0:  # and atom.GetProp('FG') == '':
            num_O, num_X, num_C, num_N, num_S = 0, 0, 0, 0, 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() in ["F", "Cl", "Br", "I"]:
                    num_X += 1
                if neighbor.GetSymbol() == "O":
                    num_O += 1
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
                if neighbor.GetSymbol() == "N":
                    num_N += 1
                if neighbor.GetSymbol() == "S":
                    num_S += 1

            if (
                num_H == 1
                and atom_num_neighbors == 3
                and charge == 0
                and atom.GetProp("FG") == ""
            ):
                atom.SetProp("FG", "tertiary_carbon")
            if atom_num_neighbors == 4 and charge == 0 and atom.GetProp("FG") == "":
                atom.SetProp("FG", "quaternary_carbon")
            if (
                num_H == 0
                and atom_num_neighbors == 3
                and charge == 0
                and atom.GetProp("FG") == ""
                and not in_ring
            ):
                atom.SetProp("FG", "alkene_carbon")

            if (
                num_O == 1
                and atom_symbol == "C"
                and atom.GetProp("FG")
                not in [
                    "hemiacetal",
                    "hemiketal",
                    "acetal",
                    "ketal",
                    "orthoester",
                    "orthocarbonate_ester",
                    "carbonate_ester",
                ]
            ):
                if num_N == 1:  # Cyanate and Isocyanate
                    condition1, condition2 = False, False
                    condition3, condition4 = False, False
                    for neighbor in atom_neighbors:
                        if (
                            neighbor.GetSymbol() == "N"
                            and mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.TRIPLE
                            and neighbor.GetFormalCharge() == 0
                        ):
                            condition1 = True
                        if (
                            neighbor.GetSymbol() == "O"
                            and mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.SINGLE
                        ):
                            condition2 = True

                        if (
                            neighbor.GetSymbol() == "N"
                            and mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.DOUBLE
                            and neighbor.GetFormalCharge() == 0
                        ):
                            condition3 = True
                        if (
                            neighbor.GetSymbol() == "O"
                            and mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.DOUBLE
                        ):
                            condition4 = True

                    if condition1 and condition2 and not in_ring:  # Cyanate
                        atom.SetProp("FG", "cyanate")
                        for neighbor in atom_neighbors:
                            neighbor.SetProp("FG", "cyanate")
                        for neighbor in atom_neighbors:
                            if neighbor.GetSymbol() == "O":
                                for C_neighbor in neighbor.GetNeighbors():
                                    if (
                                        C_neighbor.GetSymbol() in ["C", "*"]
                                        and C_neighbor.GetIdx() != atom_idx
                                    ):
                                        C_neighbor.SetProp("FG", "")

                    if condition3 and condition4 and not in_ring:  # Isocyanate
                        atom.SetProp("FG", "isocyanate")
                        for neighbor in atom_neighbors:
                            neighbor.SetProp("FG", "isocyanate")

                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() == "O":
                        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                        bondtype = bond.GetBondType()
                        if (
                            bondtype == Chem.BondType.SINGLE
                        ):  # and not neighbor.IsInRing(): # [C-O]: Alcohol (COH) or Ether [COC] or Hydroperoxy [C-O-O-H] or Peroxide [C-O-O-C]
                            if neighbor.GetTotalNumHs() == 1:  # Alcohol [COH]
                                neighbor.SetProp("FG", "hydroxyl")
                            else:
                                for O_neighbor in neighbor.GetNeighbors():
                                    # if not O_neighbor.IsInRing():
                                    if (
                                        O_neighbor.GetIdx() != atom_idx
                                        and O_neighbor.GetSymbol() in ["C", "*"]
                                        and neighbor.GetProp("FG") == ""
                                    ):  # Ether [COC]
                                        neighbor.SetProp("FG", "ether")
                                    if O_neighbor.GetSymbol() == "O":
                                        if (
                                            O_neighbor.GetTotalNumHs() == 1
                                        ):  # Hydroperoxy [C-O-O-H]
                                            neighbor.SetProp("FG", "hydroperoxy")
                                            O_neighbor.SetProp("FG", "hydroperoxy")
                                        else:
                                            neighbor.SetProp("FG", "peroxy")
                                            O_neighbor.SetProp("FG", "peroxy")

                        if (
                            bondtype == Chem.BondType.DOUBLE
                        ):  # [C=O]: Ketone [CC(=0)C] or Aldehyde [CC(=O)H] or Acyl halide [C(=O)X]
                            if (
                                num_X == 1 and not neighbor.IsInRing()
                            ):  # Acyl halide [C(=O)X]
                                atom.SetProp("FG", "haloformyl")
                                for neighbor_ in atom_neighbors:
                                    if neighbor_.GetSymbol() in [
                                        "O",
                                        "F",
                                        "Cl",
                                        "Br",
                                        "I",
                                    ]:
                                        neighbor_.SetProp("FG", "haloformyl")

                            if (
                                (num_C == 1 and num_H == 1)
                                or num_H == 2
                                and not in_ring
                            ):  # Aldehyde [C(=O)H]
                                atom.SetProp("FG", "aldehyde")
                                neighbor.SetProp("FG", "aldehyde")

                            if atom_num_neighbors == 3 and atom.GetProp("FG") not in [
                                "haloformyl",
                                "amide",
                            ]:  # Ketone [C(=0)C]
                                atom.SetProp("FG", "ketone")
                                for neighbor in atom_neighbors:
                                    if (
                                        neighbor.GetSymbol() == "O"
                                        and not neighbor.IsInRing()
                                    ):
                                        neighbor.SetProp("FG", "ketone")

            if num_O == 2:  # and atom.GetProp('FG') == '':
                if atom_num_neighbors == 3:
                    if num_H == 0:
                        condition1, condition2, condition3, condition4 = (
                            False,
                            False,
                            False,
                            False,
                        )
                        for neighbor in atom_neighbors:
                            if (
                                neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    atom.GetIdx(), neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.DOUBLE
                                and neighbor.GetFormalCharge() == 0
                                and not neighbor.IsInRing()
                            ):
                                condition1 = True
                            if (
                                neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    atom.GetIdx(), neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.SINGLE
                                and neighbor.GetFormalCharge() == -1
                                and not neighbor.IsInRing()
                            ):
                                condition2 = True
                            if (
                                neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    atom.GetIdx(), neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.SINGLE
                                and neighbor.GetFormalCharge() == 0
                                and neighbor.GetTotalNumHs() == 1
                                and not neighbor.IsInRing()
                            ):
                                condition3 = True
                            if (
                                neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    atom.GetIdx(), neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.SINGLE
                                and neighbor.GetFormalCharge() == 0
                                and neighbor.GetTotalNumHs() == 0
                                and atom.GetProp("FG") != "carbamate"
                            ):
                                condition4 = True

                        if condition1 and condition2:
                            atom.SetProp("FG", "carboxylate")
                            for neighbor in atom_neighbors:
                                if neighbor.GetSymbol() == "O":
                                    neighbor.SetProp("FG", "carboxylate")
                        if condition1 and condition3:
                            atom.SetProp("FG", "carboxyl")
                            for neighbor in atom_neighbors:
                                if neighbor.GetSymbol() == "O":
                                    neighbor.SetProp("FG", "carboxyl")
                        if (
                            condition1
                            and condition4
                            and atom.GetProp("FG")
                            not in ["carbamate", "carbonate_ester"]
                        ):
                            atom.SetProp("FG", "ester")
                            for neighbor in atom_neighbors:
                                if neighbor.GetSymbol() == "O":
                                    neighbor.SetProp("FG", "ester")
                                    for O_neighbor in neighbor.GetNeighbors():
                                        O_neighbor.SetProp("FG", "ester")

                    if num_H == 1 and not in_ring:
                        condition1, condition2 = False, False
                        cnt = 0
                        for neighbor in atom_neighbors:
                            if (
                                neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    atom.GetIdx(), neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.SINGLE
                                and neighbor.GetFormalCharge() == 0
                                and neighbor.GetTotalNumHs() == 1
                            ):
                                condition1 = True
                            if (
                                neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    atom.GetIdx(), neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.SINGLE
                                and neighbor.GetFormalCharge() == 0
                                and neighbor.GetTotalNumHs() == 0
                            ):
                                condition2 = True
                                cnt += 1

                        if condition1 and condition2:
                            atom.SetProp("FG", "hemiacetal")
                            for neighbor in atom_neighbors:
                                if neighbor.GetSymbol() == "O":
                                    neighbor.SetProp("FG", "hemiacetal")
                        if cnt == 2:
                            atom.SetProp("FG", "acetal")
                            for neighbor in atom_neighbors:
                                if neighbor.GetSymbol() == "O":
                                    neighbor.SetProp("FG", "acetal")

                if atom_num_neighbors == 4 and not in_ring:
                    condition1, condition2 = False, False
                    cnt = 0
                    for neighbor in atom_neighbors:
                        if (
                            neighbor.GetSymbol() == "O"
                            and mol.GetBondBetweenAtoms(
                                atom.GetIdx(), neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.SINGLE
                            and neighbor.GetFormalCharge() == 0
                            and neighbor.GetTotalNumHs() == 1
                            and not neighbor.IsInRing()
                        ):
                            condition1 = True
                        if (
                            neighbor.GetSymbol() == "O"
                            and mol.GetBondBetweenAtoms(
                                atom.GetIdx(), neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.SINGLE
                            and neighbor.GetFormalCharge() == 0
                            and neighbor.GetTotalNumHs() == 0
                            and not neighbor.IsInRing()
                        ):
                            condition2 = True
                            cnt += 1

                    if condition1 and condition2:
                        atom.SetProp("FG", "hemiketal")
                        for neighbor in atom_neighbors:
                            if neighbor.GetSymbol() == "O":
                                neighbor.SetProp("FG", "hemiketal")
                    if cnt == 2:
                        atom.SetProp("FG", "ketal")
                        for neighbor in atom_neighbors:
                            if neighbor.GetSymbol() == "O":
                                neighbor.SetProp("FG", "ketal")

            if num_O == 3 and atom_num_neighbors == 4 and not in_ring:
                n_C = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == 0
                        and neighbor.GetTotalNumHs() == 0
                    ):
                        n_C += 1
                if n_C == 3:
                    atom.SetProp("FG", "orthoester")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "orthoester")

            if num_O == 3 and atom_num_neighbors == 3 and charge == 0 and not in_ring:
                condition1 = False
                n_O = 0
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        mol.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == 0
                        and neighbor.GetTotalNumHs() == 0
                    ):
                        n_O += 1
                if condition1 and n_O == 2:
                    atom.SetProp("FG", "carbonate_ester")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "carbonate_ester")

            if num_O == 4 and not in_ring:
                n_C = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == 0
                        and neighbor.GetTotalNumHs() == 0
                    ):
                        n_C += 1
                if n_C == 4:
                    atom.SetProp("FG", "orthocarbonate_ester")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "orthocarbonate_ester")

            ########################### Groups containing nitrogen ###########################
            #### Amidine ####
            if num_N == 2 and atom_num_neighbors == 3:
                condition1, condition2 = False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "N"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 2
                        and neighbor.GetFormalCharge() == 0
                        and not neighbor.IsInRing()
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "N"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and len(neighbor.GetNeighbors()) == 3
                        and neighbor.GetFormalCharge() == 0
                        and not neighbor.IsInRing()
                    ):
                        condition2 = True
                if condition1 and condition2:
                    atom.SetProp("FG", "amidine")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "N":
                            neighbor.SetProp("FG", "amidine")

            if num_N == 1 and num_O == 2 and atom_num_neighbors == 3:
                condition1, condition2, condition3 = False, False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == 0
                        and len(neighbor.GetNeighbors()) == 2
                    ):
                        condition2 = True
                    if (
                        neighbor.GetSymbol() == "N"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == 0
                        and len(neighbor.GetNeighbors()) == 3
                        and not neighbor.IsInRing()
                    ):
                        condition3 = True
                if condition1 and condition2 and condition3:
                    atom.SetProp("FG", "carbamate")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "carbamate")

            if num_N == 1 and num_S == 1:
                condition1, condition2 = False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "N"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 2
                        and not neighbor.IsInRing()
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 1
                        and neighbor.GetTotalNumHs() == 0
                        and not neighbor.IsInRing()
                    ):
                        condition2 = True
                if condition1 and condition2:
                    atom.SetProp("FG", "isothiocyanate")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "isothiocyanate")

            if num_S == 1 and atom_num_neighbors == 3:
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 1
                        and neighbor.GetTotalNumHs() == 0
                        and not neighbor.IsInRing()
                    ):
                        atom.SetProp("FG", "thioketone")
                        neighbor.SetProp("FG", "thioketone")

            if num_S == 1 and num_H == 1 and atom_num_neighbors == 2:
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 1
                        and neighbor.GetTotalNumHs() == 0
                        and not neighbor.IsInRing()
                    ):
                        atom.SetProp("FG", "thial")
                        neighbor.SetProp("FG", "thial")

            if num_S == 1 and num_O == 1 and atom_num_neighbors == 3:
                condition1, condition2 = False, False
                condition3, condition4 = False, False
                condition5, condition6 = False, False
                condition7, condition8 = False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and len(neighbor.GetNeighbors()) == 1
                        and neighbor.GetTotalNumHs() == 1
                        and not neighbor.IsInRing()
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and not neighbor.IsInRing()
                    ):
                        condition2 = True

                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                        and not neighbor.IsInRing()
                    ):
                        condition3 = True
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetTotalNumHs() == 0
                        and not len(neighbor.GetNeighbors()) == 1
                    ):
                        condition4 = True

                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and len(neighbor.GetNeighbors()) == 2
                        and neighbor.GetTotalNumHs() == 0
                        and not neighbor.IsInRing()
                    ):
                        flag = True
                        for bond in neighbor.GetBonds():
                            if bond.GetBondType() != Chem.BondType.SINGLE:
                                flag = False
                        if flag:
                            condition5 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and not neighbor.IsInRing()
                    ):
                        condition6 = True

                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and len(neighbor.GetNeighbors()) == 2
                        and neighbor.GetFormalCharge() == 0
                        and not neighbor.IsInRing()
                    ):
                        condition7 = True
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetTotalNumHs() == 0
                        and len(neighbor.GetNeighbors()) == 1
                        and not neighbor.IsInRing()
                    ):
                        condition8 = True

                if condition1 and condition2:
                    atom.SetProp("FG", "carbothioic_S-acid")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["S", "O"]:
                            neighbor.SetProp("FG", "carbothioic_S-acid")
                if condition3 and condition4:
                    atom.SetProp("FG", "carbothioic_O-acid")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["S", "O"]:
                            neighbor.SetProp("FG", "carbothioic_O-acid")
                if condition5 and condition6:
                    atom.SetProp("FG", "thiolester")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["S", "O"]:
                            neighbor.SetProp("FG", "thiolester")
                if condition7 and condition8:
                    atom.SetProp("FG", "thionoester")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["S", "O"]:
                            neighbor.SetProp("FG", "thionoester")

            if num_S == 2 and atom_num_neighbors == 3:
                condition1, condition2, condition3 = False, False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                        and len(neighbor.GetNeighbors()) == 1
                        and not neighbor.IsInRing()
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetTotalNumHs() == 0
                        and len(neighbor.GetNeighbors()) == 1
                        and not neighbor.IsInRing()
                    ):
                        condition2 = True
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            neighbor.GetIdx(), atom_idx
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 0
                        and len(neighbor.GetNeighbors()) == 2
                        and not neighbor.IsInRing()
                    ):
                        flag = True
                        for bond in neighbor.GetBonds():
                            if bond.GetBondType() != Chem.BondType.SINGLE:
                                flag = False
                        if flag:
                            condition3 = True

                if condition1 and condition2:
                    atom.SetProp("FG", "carbodithioic_acid")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "S":
                            neighbor.SetProp("FG", "carbodithioic_acid")
                if condition3 and condition2:
                    atom.SetProp("FG", "carbodithio")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "S":
                            neighbor.SetProp("FG", "carbodithio")

            if num_X == 3 and charge == 0 and atom_num_neighbors == 4:
                num_F, num_Cl, num_Br, num_I = 0, 0, 0, 0
                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() == "F":
                        num_F += 1
                    if neighbor.GetSymbol() == "Cl":
                        num_Cl += 1
                    if neighbor.GetSymbol() == "Br":
                        num_Br += 1
                    if neighbor.GetSymbol() == "I":
                        num_I += 1
                if num_F == 3:
                    atom.SetProp("FG", "trifluoromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "F":
                            neighbor.SetProp("FG", "trifluoromethyl")
                if num_F == 2 and num_Cl == 1:
                    atom.SetProp("FG", "difluorochloromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["F", "Cl"]:
                            neighbor.SetProp("FG", "difluorochloromethyl")
                if num_F == 2 and num_Br == 1:
                    atom.SetProp("FG", "bromodifluoromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["F", "Br"]:
                            neighbor.SetProp("FG", "bromodifluoromethyl")

                if num_Cl == 3:
                    atom.SetProp("FG", "trichloromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "Cl":
                            neighbor.SetProp("FG", "trichloromethyl")
                if num_Cl == 2 and num_Br == 1:
                    atom.SetProp("FG", "bromodichloromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["Cl", "Br"]:
                            neighbor.SetProp("FG", "bromodichloromethyl")

                if num_Br == 3:
                    atom.SetProp("FG", "tribromomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "Br":
                            neighbor.SetProp("FG", "tribromomethyl")
                if num_Br == 2 and num_F == 1:
                    atom.SetProp("FG", "dibromofluoromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["F", "Br"]:
                            neighbor.SetProp("FG", "dibromofluoromethyl")

                if num_I == 3:
                    atom.SetProp("FG", "triiodomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "I":
                            neighbor.SetProp("FG", "triiodomethyl")

            if num_X == 2 and charge == 0 and atom_num_neighbors == 3 and num_H == 1:
                num_F, num_Cl, num_Br, num_I = 0, 0, 0, 0
                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() == "F":
                        num_F += 1
                    if neighbor.GetSymbol() == "Cl":
                        num_Cl += 1
                    if neighbor.GetSymbol() == "Br":
                        num_Br += 1
                    if neighbor.GetSymbol() == "I":
                        num_I += 1

                if num_F == 2:
                    atom.SetProp("FG", "difluoromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "F":
                            neighbor.SetProp("FG", "difluoromethyl")
                if num_F == 1 and num_Cl == 1:
                    atom.SetProp("FG", "fluorochloromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["F", "Cl"]:
                            neighbor.SetProp("FG", "fluorochloromethyl")

                if num_Cl == 2:
                    atom.SetProp("FG", "dichloromethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "Cl":
                            neighbor.SetProp("FG", "dichloromethyl")
                if num_Cl == 1 and num_Br == 1:
                    atom.SetProp("FG", "chlorobromomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["Cl", "Br"]:
                            neighbor.SetProp("FG", "chlorobromomethyl")
                if num_Cl == 1 and num_I == 1:
                    atom.SetProp("FG", "chloroiodomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["Cl", "I"]:
                            neighbor.SetProp("FG", "chloroiodomethyl")

                if num_Br == 2:
                    atom.SetProp("FG", "dibromomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "Br":
                            neighbor.SetProp("FG", "dibromomethyl")
                if num_Br == 1 and num_I == 1:
                    atom.SetProp("FG", "bromoiodomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() in ["Br", "I"]:
                            neighbor.SetProp("FG", "bromoiodomethyl")

                if num_I == 2:
                    atom.SetProp("FG", "diiodomethyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "I":
                            neighbor.SetProp("FG", "diiodomethyl")

            if (
                (atom_num_neighbors == 2 or atom_num_neighbors == 1)
                and not in_ring
                and atom.GetProp("FG") == ""
            ):
                bonds = atom.GetBonds()
                ns, nd, nt = 0, 0, 0
                for bond in bonds:
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        ns += 1
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        nd += 1
                    else:
                        nt += 1
                if ns >= 1 and nd == 0 and nt == 0:
                    atom.SetProp("FG", "alkyl")
                if nd >= 1:
                    atom.SetProp("FG", "alkene")
                if nt == 1:
                    atom.SetProp("FG", "alkyne")

        elif (
            atom_symbol == "O" and not in_ring and charge == 0 and num_H == 0
        ):  # Carboxylic anhydride [C(CO)O(CO)C]
            num_C = 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
            if num_C == 2:
                cnt = 0
                for neighbor in atom_neighbors:
                    for C_neighbor in neighbor.GetNeighbors():
                        if (
                            C_neighbor.GetSymbol() == "O"
                            and mol.GetBondBetweenAtoms(
                                neighbor.GetIdx(), C_neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.DOUBLE
                            and len(neighbor.GetNeighbors()) == 3
                        ):
                            cnt += 1
                if cnt == 2:
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "carboxylic_anhydride")
                        for C_neighbor in neighbor.GetNeighbors():
                            if C_neighbor.GetSymbol() == "O":
                                C_neighbor.SetProp("FG", "carboxylic_anhydride")

        elif atom_symbol == "N":  # and atom.GetProp('FG') == '':
            num_C, num_O, num_N = 0, 0, 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
                if neighbor.GetSymbol() == "O":
                    num_O += 1
                if neighbor.GetSymbol() == "N":
                    num_N += 1

            #### Amines ####
            if (
                charge == 0
                and num_H == 2
                and atom_num_neighbors == 1
                and atom.GetProp("FG") != "hydrazone"
            ):  # Primary amine [RNH2]
                atom.SetProp("FG", "primary_amine")

            if (
                charge == 0 and num_H == 1 and atom_num_neighbors == 2
            ):  # Secondary amine [R'R"NH]
                atom.SetProp("FG", "secondary_amine")

            if (
                charge == 0
                and atom_num_neighbors == 3
                and atom.GetProp("FG") != "carbamate"
            ):
                cnt = 0
                C_idx = []
                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() in ["C", "*"]:
                        for C_neighbor in neighbor.GetNeighbors():
                            if (
                                C_neighbor.GetSymbol() == "O"
                                and mol.GetBondBetweenAtoms(
                                    neighbor.GetIdx(), C_neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.DOUBLE
                                and len(neighbor.GetNeighbors()) == 3
                                and neighbor.GetFormalCharge() == 0
                                and atom.GetProp("FG") != "imide"
                            ):
                                atom.SetProp("FG", "amide")
                                neighbor.SetProp("FG", "amide")
                                C_neighbor.SetProp("FG", "amide")
                                cnt += 1
                                C_idx.append(neighbor.GetIdx())

                if cnt == 2:
                    for neighbor in atom_neighbors:
                        if neighbor.GetIdx() in C_idx:
                            for C_neighbor in neighbor.GetNeighbors():
                                if C_neighbor.GetSymbol() in ["O", "N"]:
                                    neighbor.SetProp("FG", "imide")
                                    C_neighbor.SetProp("FG", "imide")

                if atom.GetProp("FG") not in [
                    "imide",
                    "amide",
                    "amidine",
                    "carbamate",
                ]:  # Tertiary amine [R3N]
                    atom.SetProp("FG", "tertiary_amine")

            if charge == 1 and atom_num_neighbors == 4:  # 4° ammonium ion [R3N]
                atom.SetProp("FG", "4_ammonium_ion")

            if (
                charge == 0
                and num_C == 1
                and num_N == 1
                and num_H == 0
                and atom_num_neighbors == 2
            ):  # Hydrazone [R'R"CN2H2]
                condition1, condition2 = False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 3
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "N"
                        and neighbor.GetTotalNumHs() == 2
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition2 = True
                if condition1 and condition2:
                    atom.SetProp("FG", "hydrazone")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "hydrazone")

            #### Imine ####
            if (
                charge == 0
                and num_C == 1
                and num_H == 1
                and num_N == 0
                and atom_num_neighbors == 1
            ):  # Primary ketimine [RC(=NH)R']
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 3
                        and neighbor.GetFormalCharge() == 0
                    ):
                        atom.SetProp("FG", "primary_ketimine")
                        for neighbor in atom_neighbors:
                            neighbor.SetProp("FG", "primary_ketimine")

                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 2
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        atom.SetProp("FG", "primary_aldimine")
                        for neighbor in atom_neighbors:
                            neighbor.SetProp("FG", "primary_aldimine")

            if (
                charge == 0
                and atom_num_neighbors == 1
                and atom.GetProp("FG") not in ["thiocyanate", "cyanate"]
            ):  # Nitrile
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.TRIPLE
                    ):
                        atom.SetProp("FG", "nitrile")

            if (
                charge == 0
                and num_C >= 1
                and atom_num_neighbors == 2
                and atom.GetProp("FG") != "hydrazone"
            ):  # Secondary ketimine [RC(=NR'')R']
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 3
                        and neighbor.GetFormalCharge() == 0
                    ):
                        atom.SetProp("FG", "secondary_ketimine")
                        for neighbor in atom_neighbors:
                            if (
                                mol.GetBondBetweenAtoms(
                                    atom_idx, neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.DOUBLE
                            ):
                                neighbor.SetProp("FG", "secondary_ketimine")

                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and len(neighbor.GetNeighbors()) == 2
                        and neighbor.GetFormalCharge() == 0
                        and neighbor.GetTotalNumHs() == 1
                    ):
                        atom.SetProp("FG", "secondary_aldimine")
                        for neighbor in atom_neighbors:
                            if (
                                mol.GetBondBetweenAtoms(
                                    atom_idx, neighbor.GetIdx()
                                ).GetBondType()
                                == Chem.BondType.DOUBLE
                            ):
                                neighbor.SetProp("FG", "secondary_aldimine")

            if charge == 1 and num_N == 2 and atom_num_neighbors == 2:  # Azide [RN3]
                condition1, condition2 = False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetFormalCharge() == 0
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                    ):
                        condition1 = True
                    if (
                        neighbor.GetFormalCharge() == -1
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                    ):
                        condition2 = True
                if condition1 and condition2 and not in_ring:
                    atom.SetProp("FG", "azide")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "azide")

            if (
                charge == 0 and num_N == 1 and atom_num_neighbors == 2 and not in_ring
            ):  # Azo [RN2R']
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "N"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        atom.SetProp("FG", "azo")
                        neighbor.SetProp("FG", "azo")
                        break

            if (
                charge == 1 and num_O == 3 and atom_num_neighbors == 3
            ):  # Nitrate [RONO2]
                condition1, condition2, condition3 = False, False, False
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == -1
                    ):
                        condition2 = True
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition3 = True

                if condition1 and condition2 and condition3 and not in_ring:
                    atom.SetProp("FG", "nitrate")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "nitrate")

            if charge == 1 and num_C >= 1 and atom_num_neighbors == 2:  # Isonitrile
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and neighbor.GetFormalCharge() == -1
                        and len(neighbor.GetNeighbors()) == 1
                    ):
                        atom.SetProp("FG", "isonitrile")
                        neighbor.SetProp("FG", "isonitrile")

            if (
                charge == 0 and num_O == 2 and atom_num_neighbors == 2 and not in_ring
            ):  # Nitrite
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and len(neighbor.GetNeighbors()) == 2
                    ):
                        atom.SetProp("FG", "nitrosooxy")
                        for neighbor in atom_neighbors:
                            neighbor.SetProp("FG", "nitrosooxy")

            if (
                charge == 1 and num_O == 2 and atom_num_neighbors == 3 and not in_ring
            ):  # Nitro compound
                condition1, condition2 = False, False
                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() == "O":
                        if (
                            mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.DOUBLE
                            and neighbor.GetFormalCharge() == 0
                        ):
                            condition1 = True
                        if (
                            mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.SINGLE
                            and neighbor.GetFormalCharge() == -1
                        ):
                            condition2 = True
                if condition1 and condition2 and not in_ring:
                    atom.SetProp("FG", "nitro")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "nitro")

            if charge == 0 and num_O == 1 and atom_num_neighbors == 2 and not in_ring:
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                    ):  # Nitroso compound
                        atom.SetProp("FG", "nitroso")
                        neighbor.SetProp("FG", "nitroso")

            if charge == 0 and num_O == 1 and num_C == 1 and atom_num_neighbors == 2:
                condition1, condition2, condition3 = False, False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition2 = True
                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and neighbor.GetTotalNumHs() == 0
                        and neighbor.GetFormalCharge() == 0
                        and len(neighbor.GetNeighbors()) == 3
                    ):
                        condition3 = True

                if condition1 and condition2 and not in_ring:
                    atom.SetProp("FG", "aldoxime")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "aldoxime")
                if condition1 and condition3 and not in_ring:
                    atom.SetProp("FG", "ketoxime")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "ketoxime")

        ########################### Groups containing sulfur ###########################
        elif atom_symbol == "S" and charge == 0:
            num_C, num_S, num_O = 0, 0, 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
                if neighbor.GetSymbol() == "S":
                    num_S += 1
                if neighbor.GetSymbol() == "O":
                    num_O += 1

            if (
                num_H == 1
                and atom_num_neighbors == 1
                and atom.GetProp("FG")
                not in ["carbothioic_S-acid", "carbodithioic_acid"]
            ):
                neighbor = atom_neighbors[0]
                if (
                    mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx()).GetBondType()
                    == Chem.BondType.SINGLE
                ):
                    atom.SetProp("FG", "sulfhydryl")

            if (
                num_H == 0
                and atom_num_neighbors == 2
                and atom.GetProp("FG") not in ["sulfhydrylester", "carbodithio"]
            ):
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        cnt += 1
                if cnt == 2:
                    atom.SetProp("FG", "sulfide")

            if num_H == 0 and num_S == 1 and atom_num_neighbors == 2:
                condition1, condition2 = False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "S"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and len(neighbor.GetNeighbors()) == 2
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() != "S"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        condition2 = True
                if condition1 and condition2:
                    atom.SetProp("FG", "disulfide")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "S":
                            neighbor.SetProp("FG", "disulfide")

            if num_H == 0 and num_O >= 1 and atom_num_neighbors == 3:
                condition = False
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition = True
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        cnt += 1
                if condition and cnt == 2:
                    atom.SetProp("FG", "sulfinyl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "sulfinyl")

            if num_H == 0 and num_O >= 2 and atom_num_neighbors == 4:
                cnt1 = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                    ):
                        cnt1 += 1
                if cnt1 == 2:
                    atom.SetProp("FG", "sulfonyl")
                    for neighbor in atom_neighbors:
                        if (
                            neighbor.GetSymbol() == "O"
                            and mol.GetBondBetweenAtoms(
                                atom_idx, neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.DOUBLE
                        ):
                            neighbor.SetProp("FG", "sulfonyl")

            if num_H == 0 and num_O == 2 and atom_num_neighbors == 3:
                condition1, condition2, condition3 = False, False, False
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition2 = True
                    if (
                        neighbor.GetSymbol() != "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        condition3 = True
                if condition1 and condition2 and condition3 and not in_ring:
                    atom.SetProp("FG", "sulfino")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "sulfino")

            if num_H == 0 and num_O == 3 and atom_num_neighbors == 4:
                condition1, condition2 = False, False
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        cnt += 1
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() != "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        condition2 = True
                if condition1 and condition2 and cnt == 2 and not in_ring:
                    atom.SetProp("FG", "sulfonic_acid")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "sulfonic_acid")

            if num_H == 0 and num_O == 3 and atom_num_neighbors == 4:
                condition1, condition2 = False, False
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        cnt += 1
                    if (
                        neighbor.GetSymbol() != "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 0
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition2 = True
                if condition1 and condition2 and cnt == 2:
                    atom.SetProp("FG", "sulfonate_ester")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "sulfonate_ester")

            if num_H == 0 and atom_num_neighbors == 2:
                for neighbor in atom_neighbors:
                    for C_neighbor in neighbor.GetNeighbors():
                        if (
                            C_neighbor.GetSymbol() == "N"
                            and mol.GetBondBetweenAtoms(
                                C_neighbor.GetIdx(), neighbor.GetIdx()
                            ).GetBondType()
                            == Chem.BondType.TRIPLE
                            and not in_ring
                        ):
                            atom.SetProp("FG", "thiocyanate")
                            neighbor.SetProp("FG", "thiocyanate")
                            C_neighbor.SetProp("FG", "thiocyanate")

        ########################### Groups containing phosphorus ###########################
        elif atom_symbol == "P" and not in_ring and charge == 0:
            num_C, num_O = 0, 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
                if neighbor.GetSymbol() == "O":
                    num_O += 1

            if atom_num_neighbors == 3:
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        cnt += 1
                if cnt == 3:
                    atom.SetProp("FG", "phosphino")

            if num_O == 3 and atom_num_neighbors == 4:
                condition1, condition2 = False, False
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        cnt += 1
                    if (
                        neighbor.GetSymbol() != "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        condition2 = True
                if condition1 and condition2 and cnt == 2:
                    atom.SetProp("FG", "phosphono")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "phosphono")

            if num_O == 4 and atom_num_neighbors == 4:
                condition1 = False
                cnt1, cnt2 = 0, 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                    ):
                        condition1 = True
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        cnt1 += 1
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                        and neighbor.GetTotalNumHs() == 0
                        and neighbor.GetFormalCharge() == 0
                    ):
                        cnt2 += 1

                if condition1 and cnt1 == 2 and cnt2 == 1:
                    atom.SetProp("FG", "phosphate")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "phosphate")
                if condition1 and cnt1 == 1 and cnt2 == 2:
                    atom.SetProp("FG", "phosphodiester")
                    for neighbor in atom_neighbors:
                        neighbor.SetProp("FG", "phosphodiester")

            if num_O == 1 and atom_num_neighbors == 4:
                condition = False
                cnt = 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.DOUBLE
                        and neighbor.GetFormalCharge() == 0
                    ):
                        condition = True
                    if (
                        mol.GetBondBetweenAtoms(
                            atom_idx, neighbor.GetIdx()
                        ).GetBondType()
                        == Chem.BondType.SINGLE
                    ):
                        cnt += 1
                if condition and cnt == 3:
                    atom.SetProp("FG", "phosphoryl")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "phosphoryl")

        ########################### Groups containing boron ###########################
        elif atom_symbol == "B" and not in_ring and charge == 0:
            num_C, num_O = 0, 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
                if neighbor.GetSymbol() == "O":
                    num_O += 1

            if num_O == 2 and atom_num_neighbors == 3:
                cnt1, cnt2 = 0, 0
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and neighbor.GetTotalNumHs() == 1
                        and neighbor.GetFormalCharge() == 0
                    ):
                        cnt1 += 1
                    if (
                        neighbor.GetSymbol() == "O"
                        and neighbor.GetFormalCharge() == 0
                        and len(neighbor.GetNeighbors()) == 2
                    ):
                        cnt2 += 1
                if cnt1 == 2:
                    atom.SetProp("FG", "borono")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "borono")
                if cnt2 == 2:
                    atom.SetProp("FG", "boronate")
                    for neighbor in atom_neighbors:
                        if neighbor.GetSymbol() == "O":
                            neighbor.SetProp("FG", "boronate")

            if num_O == 1 and atom_num_neighbors == 3:
                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() == "O" and neighbor.GetFormalCharge() == 0:
                        if neighbor.GetTotalNumHs() == 1:
                            atom.SetProp("FG", "borino")
                            neighbor.SetProp("FG", "borino")
                        if len(neighbor.GetNeighbors()) == 2:
                            atom.SetProp("FG", "borinate")
                            neighbor.SetProp("FG", "borinate")

        ########################### Groups containing silicon ###########################
        elif atom_symbol == "Si" and not in_ring and charge == 0:
            num_O, num_Cl, num_C = 0, 0, 0
            for neighbor in atom_neighbors:
                if neighbor.GetSymbol() == "O":
                    num_O += 1
                if neighbor.GetSymbol() == "Cl":
                    num_Cl += 1
                if neighbor.GetSymbol() in ["C", "*"]:
                    num_C += 1
            if num_O == 1 and charge == 0 and atom_num_neighbors == 4:
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() == "O"
                        and len(neighbor.GetNeighbors()) == 2
                        and neighbor.GetFormalCharge() == 0
                    ):
                        atom.SetProp("FG", "silyl_ether")
                        neighbor.SetProp("FG", "silyl_ether")
            if num_Cl == 2 and charge == 0 and atom_num_neighbors == 4:
                for neighbor in atom_neighbors:
                    if neighbor.GetSymbol() == "Cl" and neighbor.GetFormalCharge() == 0:
                        atom.SetProp("FG", "dichlorosilane")
                        neighbor.SetProp("FG", "dichlorosilane")
            if (
                num_C >= 3
                and charge == 0
                and atom_num_neighbors == 4
                and atom.GetProp("FG") != "silyl_ether"
            ):
                cnt = 0
                C_idx = []
                for neighbor in atom_neighbors:
                    if (
                        neighbor.GetSymbol() in ["C", "*"]
                        and neighbor.GetFormalCharge() == 0
                        and neighbor.GetTotalNumHs() == 3
                    ):
                        cnt += 1
                        C_idx.append(neighbor.GetIdx())
                if cnt == 3:
                    atom.SetProp("FG", "trimethylsilyl")
                    for idx in C_idx:
                        mol.GetAtomWithIdx(idx).SetProp("FG", "trimethylsilyl")

        ########################### Groups containing halogen ###########################
        elif (
            atom_symbol == "F"
            and not in_ring
            and charge == 0
            and atom.GetProp("FG") == ""
        ):
            atom.SetProp("FG", "fluoro")
        elif (
            atom_symbol == "Cl"
            and not in_ring
            and charge == 0
            and atom.GetProp("FG") == ""
        ):
            atom.SetProp("FG", "chloro")
        elif (
            atom_symbol == "Br"
            and not in_ring
            and charge == 0
            and atom.GetProp("FG") == ""
        ):
            atom.SetProp("FG", "bromo")
        elif (
            atom_symbol == "I"
            and not in_ring
            and charge == 0
            and atom.GetProp("FG") == ""
        ):
            atom.SetProp("FG", "iodo")
        else:
            pass

        ########################### Groups containing other elements ###########################
        if atom.GetProp("FG") == "" and atom_symbol in ELEMENTS and not in_ring:
            if charge == 0:
                atom.SetProp("FG", atom_symbol)
                atom.SetProp(FLAG_NO_FG, "")
                # atom.SetProp("FG", NO_FG) # changes the fg-detection algo (num of fg dectected reduces)
            else:
                atom.SetProp("FG", f"{atom_symbol}[{charge}]")
                atom.SetProp(FLAG_NO_FG, "")
                # atom.SetProp("FG", NO_FG) # changes the fg-detection algo (num of fg dectected reduces)
        else:
            pass

        if atom_symbol == "*":
            atom.SetProp("FG", "")


def set_atom_map_num(mol):
    if mol is not None:
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx != 0:
                atom.SetAtomMapNum(idx)
            else:
                atom.SetAtomMapNum(-9)


def find_neighbor_map(smiles):
    matches = re.findall(r"\[(\d+)\*\]", smiles)
    idx = [int(match) for match in matches]
    if smiles.startswith("*"):
        return set(idx) | {0}
    else:
        return set(idx)


def find_atom_map(smiles):
    matches = re.findall(r"\[[^\]]*:(\d+)\]", smiles)
    idx = [int(match) for match in matches]
    return set(idx)


def get_structure(mol):
    set_atom_map_num(mol)
    fused_rings_groups: list[list[set[int]]] = set_ring_properties(mol)
    detect_functional_group(mol)
    rings = mol.GetRingInfo().AtomRings()

    splitting_bonds = set()
    for bond in mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_atom_prop = begin_atom.GetProp("FG")
        end_atom_prop = end_atom.GetProp("FG")
        begin_atom_symbol = begin_atom.GetSymbol()
        end_atom_symbol = end_atom.GetSymbol()

        if (
            (begin_atom.IsInRing() and not end_atom.IsInRing())
            or (not begin_atom.IsInRing() and end_atom.IsInRing())
            or (begin_atom.IsInRing() and end_atom.IsInRing())
        ):
            flag = True
            for ring in rings:
                if begin_atom.GetIdx() in ring and end_atom.GetIdx() in ring:
                    flag = False
                    break
            if flag:
                splitting_bonds.add(bond)
        else:
            if begin_atom_prop != end_atom_prop:
                splitting_bonds.add(bond)
            if begin_atom_prop == "" and end_atom_prop == "":
                if (begin_atom_symbol in ["C", "*"] and end_atom_symbol != "C") or (
                    begin_atom_symbol != "C" and end_atom_symbol in ["C", "*"]
                ):
                    splitting_bonds.add(bond)

    splitting_bonds = list(splitting_bonds)
    if splitting_bonds != []:
        fragments = Chem.FragmentOnBonds(
            mol, [bond.GetIdx() for bond in splitting_bonds], addDummies=True
        )
        BONDS = set()
        for bond in splitting_bonds:
            BONDS.add(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
            )
    else:
        fragments = mol
        BONDS = set()
    smiles = m2s(fragments).replace("-9", "0").split(".")

    structure = {}
    for frag in smiles:
        atom_idx: set = find_atom_map(frag)
        structure[frag] = {"atom": atom_idx, "is_ring_fg": False}

        # Convert fragment SMILES back to mol to match with fused ring atom indices
        frag_mol = Chem.MolFromSmiles(frag)
        frag_rings = frag_mol.GetRingInfo().AtomRings()
        if len(frag_rings) >= 1:
            structure[frag]["is_ring_fg"] = True

        if len(frag_rings) <= 1:
            continue

        if not fused_rings_groups:
            continue

        for group in fused_rings_groups:
            flat_atoms = set().union(*group)
            if flat_atoms.issubset(atom_idx) and len(flat_atoms) == len(atom_idx):
                for idx, ring_atoms in enumerate(group):
                    structure[f"{frag}_{idx + 1}"] = {
                        "atom": ring_atoms,
                        "is_ring_fg": True,
                    }
                structure.pop(frag)
                break

    return structure, BONDS
