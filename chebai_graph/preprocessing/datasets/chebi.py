from chebai.preprocessing.datasets.chebi import ChEBIOver50

from chebai_graph.preprocessing.reader import GraphReader, GraphPropertyReader
from chebai_graph.preprocessing.property_encoder import IndexEncoder
from chebai_graph.preprocessing.properties import (
    AtomProperty,
    BondProperty,
    MolecularProperty,
)
import pandas as pd
from torch_geometric.data.data import Data as GeomData
import torch
import chebai_graph.preprocessing.properties as graph_properties
import importlib
import os
import tqdm
import numpy as np


class ChEBI50GraphData(ChEBIOver50):
    READER = GraphReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _resolve_property(
    property,  #: str | properties.MolecularProperty
) -> MolecularProperty:
    # split class_path into module-part and class name
    if isinstance(property, MolecularProperty):
        return property
    try:
        last_dot = property.rindex(".")
        module_name = property[:last_dot]
        class_name = property[last_dot + 1 :]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
    except ValueError:
        # if only a class name is given, assume the module is chebai_graph.processing.properties
        return getattr(graph_properties, property)()


class ChEBI50GraphProperties(ChEBIOver50):
    READER = GraphPropertyReader

    def __init__(self, atom_properties, bond_properties, **kwargs):
        super().__init__(**kwargs)
        # atom_properties and bond_properties are given as lists containing class_paths
        if atom_properties is not None:
            atom_properties = [_resolve_property(prop) for prop in atom_properties]
        else:
            atom_properties = []
        if bond_properties is not None:
            bond_properties = [_resolve_property(prop) for prop in bond_properties]
        else:
            atom_properties = []
        self.atom_properties = atom_properties
        self.bond_properties = bond_properties

    def _setup_properties(self):
        raw_data = []
        os.makedirs(self.processed_atom_properties_dir, exist_ok=True)
        os.makedirs(self.processed_bond_properties_dir, exist_ok=True)

        for raw_file in self.raw_file_names:
            path = os.path.join(self.raw_dir, raw_file)
            raw_data += list(self._load_dict(path))
        idents = [row["ident"] for row in raw_data]
        features = [row["features"] for row in raw_data]

        # use vectorized version of encode function, apply only if value is present
        enc_if_not_none = (
            lambda f, v: torch.tensor(f(v.numpy()))
            if v is not None and len(v) > 0
            else None
        )

        for property in self.atom_properties:
            assert isinstance(property, AtomProperty)
            if not os.path.isfile(self.get_atom_property_path(property)):
                print(f"Process atom property {property.name}")
                # read all property values first, then encode
                property_values = [
                    self.reader.read_atom_property(feat, property)
                    for feat in tqdm.tqdm(features)
                ]
                enc_vec = np.vectorize(property.encoder.encode)
                encoded_values = torch.cat(
                    [
                        enc_if_not_none(enc_vec, value)
                        for value in tqdm.tqdm(property_values)
                    ]
                )
                torch.save(
                    [
                        {property.name: feat, "ident": id}
                        for feat, id in zip(encoded_values, idents)
                        if feat is not None
                    ],
                    self.get_atom_property_path(property),
                )
                property.on_finish()

        for property in self.bond_properties:
            assert isinstance(property, BondProperty)
            if not os.path.isfile(self.get_bond_property_path(property)):
                print(f"Process bond property {property.name}")
                # read all property values first, then encode
                property_values = [
                    self.reader.read_bond_property(feat, property)
                    for feat in tqdm.tqdm(features)
                ]
                enc_vec = np.vectorize(property.encoder.encode)
                encoded_values = torch.cat(
                    [
                        enc_if_not_none(enc_vec, value)
                        for value in tqdm.tqdm(property_values)
                    ]
                )
                torch.save(
                    [
                        {property.name: feat, "ident": id}
                        for feat, id in zip(encoded_values, idents)
                        if feat is not None
                    ],
                    self.get_bond_property_path(property),
                )
                property.on_finish()

    @property
    def processed_atom_properties_dir(self):
        return os.path.join(self.processed_dir, "atom_properties")

    def get_atom_property_path(self, property: AtomProperty):
        return os.path.join(self.processed_atom_properties_dir, f"{property.name}.pt")

    @property
    def processed_bond_properties_dir(self):
        return os.path.join(self.processed_dir, "bond_properties")

    def get_bond_property_path(self, property: BondProperty):
        return os.path.join(self.processed_bond_properties_dir, f"{property.name}.pt")

    def setup(self, **kwargs):
        super().setup(keep_reader=True, **kwargs)
        self._setup_properties()

        self.reader.on_finish()

    @staticmethod
    def _merge_atom_prop_into_base(row, property: AtomProperty):
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData)
        x = torch.cat([geom_data.x, row[f"atom_{property.name}"].unsqueeze(1)], dim=1)
        return GeomData(
            x=x, edge_index=geom_data.edge_index, edge_attr=geom_data.edge_attr
        )

    @staticmethod
    def _merge_bond_prop_into_base(row, property: BondProperty):
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData)
        if isinstance(row[f"bond_{property.name}"], torch.Tensor):
            property_values = row[f"bond_{property.name}"].unsqueeze(1)
        else:
            property_values = torch.zeros((0, property.encoder.get_encoding_length()))
        edge_attr = torch.cat([geom_data.edge_attr, property_values], dim=1)
        return GeomData(
            x=geom_data.x, edge_index=geom_data.edge_index, edge_attr=edge_attr
        )

    def _load_processed_data(self, kind):
        """Combine base data set with property values for atoms and bonds."""
        base_filename = self.processed_file_names_dict[kind]
        base_data = torch.load(os.path.join(self.processed_dir, base_filename))
        base_df = pd.DataFrame(base_data)
        for property in self.atom_properties:
            property_data = torch.load(self.get_atom_property_path(property))
            property_df = pd.DataFrame(property_data)
            property_df.rename(
                columns={property.name: f"atom_{property.name}"}, inplace=True
            )
            base_df = base_df.merge(property_df, on="ident", how="left")
            base_df["features"] = base_df.apply(
                lambda row: self._merge_atom_prop_into_base(row, property), axis=1
            )
        for property in self.bond_properties:
            property_data = torch.load(self.get_bond_property_path(property))
            property_df = pd.DataFrame(property_data)
            property_df.rename(
                columns={property.name: f"bond_{property.name}"}, inplace=True
            )
            base_df = base_df.merge(property_df, on="ident", how="left")
            base_df["features"] = base_df.apply(
                lambda row: self._merge_bond_prop_into_base(row, property), axis=1
            )

        return base_df[base_data[0].keys()].to_dict("records")
