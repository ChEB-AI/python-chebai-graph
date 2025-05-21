import importlib
import os
from typing import Callable, List, Optional

import pandas as pd
import torch
import tqdm
from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverXPartial,
)
from lightning_utilities.core.rank_zero import rank_zero_info
from torch_geometric.data.data import Data as GeomData

import chebai_graph.preprocessing.properties as graph_properties
from chebai_graph.preprocessing.properties import (
    AtomProperty,
    BondProperty,
    MolecularProperty,
)
from chebai_graph.preprocessing.reader import (
    GraphFGAugmentorReader,
    GraphPropertyReader,
    GraphReader,
)


class ChEBI50GraphData(ChEBIOver50):
    READER = GraphReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def _resolve_property(
    property,  #: str | properties.MolecularProperty
) -> MolecularProperty:
    # if property is given as a string, try to resolve as a class path
    if isinstance(property, MolecularProperty):
        return property
    try:
        # split class_path into module-part and class name
        last_dot = property.rindex(".")
        module_name = property[:last_dot]
        class_name = property[last_dot + 1 :]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
    except ValueError:
        # if only a class name is given, assume the module is chebai_graph.processing.properties
        return getattr(graph_properties, property)()


class GraphPropertiesMixIn(XYBaseDataModule):
    READER = GraphPropertyReader

    def __init__(
        self, properties: Optional[List], transform: Optional[Callable] = None, **kwargs
    ):
        super().__init__(**kwargs)
        # atom_properties and bond_properties are given as lists containing class_paths
        if properties is not None:
            properties = [_resolve_property(prop) for prop in properties]
            properties = sorted(
                properties, key=lambda prop: self.get_property_path(prop)
            )
        else:
            properties = []
        self.properties = properties
        assert isinstance(self.properties, list) and all(
            isinstance(p, MolecularProperty) for p in self.properties
        )
        rank_zero_info(
            f"Data module uses these properties (ordered): {', '.join([str(p) for p in properties])}"
        )
        self.transform = transform

    def _setup_properties(self):
        raw_data = []
        os.makedirs(self.processed_properties_dir, exist_ok=True)

        try:
            file_names = self.processed_main_file_names
        except NotImplementedError:
            file_names = self.raw_file_names

        for file in file_names:
            # processed_dir_main only exists for ChEBI datasets
            path = os.path.join(
                (
                    self.processed_dir_main
                    if hasattr(self, "processed_dir_main")
                    else self.raw_dir
                ),
                file,
            )
            raw_data += list(self._load_dict(path))
        idents = [row["ident"] for row in raw_data]
        features = [row["features"] for row in raw_data]

        # use vectorized version of encode function, apply only if value is present
        enc_if_not_none = lambda encode, value: (
            [encode(atom_v) for atom_v in value]
            if value is not None and len(value) > 0
            else None
        )

        for property in self.properties:
            if not os.path.isfile(self.get_property_path(property)):
                rank_zero_info(f"Processing property {property.name}")
                # read all property values first, then encode
                property_values = [
                    self.reader.read_property(feat, property)
                    for feat in tqdm.tqdm(features)
                ]

                property.encoder.on_start(property_values=property_values)
                encoded_values = [
                    enc_if_not_none(property.encoder.encode, value)
                    for value in tqdm.tqdm(property_values)
                ]

                torch.save(
                    [
                        {property.name: torch.cat(feat), "ident": id}
                        for feat, id in zip(encoded_values, idents)
                        if feat is not None
                    ],
                    self.get_property_path(property),
                )
                property.on_finish()

    @property
    def processed_properties_dir(self):
        return os.path.join(self.processed_dir, "properties")

    def get_property_path(self, property: MolecularProperty):
        return os.path.join(
            self.processed_properties_dir,
            f"{property.name}_{property.encoder.name}.pt",
        )

    def setup(self, **kwargs):
        super().setup(keep_reader=True, **kwargs)
        self._setup_properties()

        self.reader.on_finish()

    def _merge_props_into_base(self, row):
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData)
        edge_attr = geom_data.edge_attr
        x = geom_data.x
        molecule_attr = torch.empty((1, 0))
        for property in self.properties:
            property_values = row[f"{property.name}"]
            if isinstance(property_values, torch.Tensor):
                if len(property_values.size()) == 0:
                    property_values = property_values.unsqueeze(0)
                if len(property_values.size()) == 1:
                    property_values = property_values.unsqueeze(1)
            else:
                property_values = torch.zeros(
                    (0, property.encoder.get_encoding_length())
                )
            if isinstance(property, AtomProperty):
                x = torch.cat([x, property_values], dim=1)
            elif isinstance(property, BondProperty):
                edge_attr = torch.cat([edge_attr, property_values], dim=1)
            else:
                molecule_attr = torch.cat([molecule_attr, property_values], dim=1)
        return GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
            molecule_attr=molecule_attr,
        )

    def load_processed_data_from_file(self, filename):
        base_data = super().load_processed_data_from_file(filename)
        base_df = pd.DataFrame(base_data)
        for property in self.properties:
            property_data = torch.load(
                self.get_property_path(property), weights_only=False
            )
            if len(property_data[0][property.name].shape) > 1:
                property.encoder.set_encoding_length(
                    property_data[0][property.name].shape[1]
                )

            property_df = pd.DataFrame(property_data)
            property_df.rename(
                columns={property.name: f"{property.name}"}, inplace=True
            )
            base_df = base_df.merge(property_df, on="ident", how="left")

        base_df["features"] = base_df.apply(
            lambda row: self._merge_props_into_base(row), axis=1
        )

        # apply transformation, e.g. masking for pretraining task
        if self.transform is not None:
            base_df["features"] = base_df["features"].apply(self.transform)

        prop_lengths = [
            (prop.name, prop.encoder.get_encoding_length()) for prop in self.properties
        ]

        rank_zero_info(
            f"Finished loading dataset from properties."
            f"\nEncoding lengths are: "
            f"{prop_lengths}"
            f"\nIf you train a model with these properties and encodings, "
            f"use n_atom_properties: {sum([prop.encoder.get_encoding_length() for prop in self.properties if isinstance(prop, AtomProperty)])}, "
            f"n_bond_properties: {sum([prop.encoder.get_encoding_length() for prop in self.properties if isinstance(prop, BondProperty)])} "
            f"and n_molecule_properties: {sum([prop.encoder.get_encoding_length() for prop in self.properties if not (isinstance(prop, AtomProperty) or isinstance(prop, BondProperty))])}"
        )

        return base_df[base_data[0].keys()].to_dict("records")


class ChEBI50GraphProperties(GraphPropertiesMixIn, ChEBIOver50):
    pass


class ChEBI100GraphProperties(GraphPropertiesMixIn, ChEBIOver100):
    pass


class ChEBI50GraphPropertiesPartial(ChEBI50GraphProperties, ChEBIOverXPartial):
    pass


class ChEBI50GraphFGAugmentorReader(GraphPropertiesMixIn, ChEBIOver50):
    READER = GraphFGAugmentorReader
