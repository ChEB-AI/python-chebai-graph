import os
from abc import ABC
from collections.abc import Callable
from pprint import pformat
from typing import Optional

import pandas as pd
import torch
import tqdm
from chebai.preprocessing.datasets.chebi import (
    ChEBIOver50,
    ChEBIOver100,
    ChEBIOverX,
    ChEBIOverXPartial,
)
from lightning_utilities.core.rank_zero import rank_zero_info
from torch_geometric.data.data import Data as GeomData

from chebai_graph.preprocessing.properties import (
    AllNodeTypeProperty,
    AtomNodeTypeProperty,
    AtomProperty,
    BondProperty,
    FGNodeTypeProperty,
    MolecularProperty,
    MoleculeProperty,
)
from chebai_graph.preprocessing.reader import (
    AtomFGReader_NoFGEdges_WithGraphNode,
    AtomFGReader_WithFGEdges_NoGraphNode,
    AtomFGReader_WithFGEdges_WithGraphNode,
    AtomReader_WithGraphNodeOnly,
    AtomsFGReader_NoFGEdges_NoGraphNode,
    GN_WithAllNodes_FG_WithAtoms_FGE,
    GN_WithAllNodes_FG_WithAtoms_NoFGE,
    GN_WithAtoms_FG_WithAtoms_FGE,
    GN_WithAtoms_FG_WithAtoms_NoFGE,
    GraphPropertyReader,
    GraphReader,
    RandomFeatureInitializationReader,
)

from .utils import resolve_property


class ChEBI50GraphData(ChEBIOver50):
    """ChEBI dataset with at least 50 samples per class, using GraphReader."""

    READER = GraphReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DataPropertiesSetter(ChEBIOverX, ABC):
    """Mixin for adding molecular property encodings to graph-based ChEBI datasets."""

    READER = GraphPropertyReader

    def __init__(
        self,
        properties: list | None = None,
        transform: Callable | None = None,
        **kwargs,
    ):
        """
        Initialize GraphPropertiesMixIn.

        Args:
            properties: Optional list of MolecularProperty class paths or instances.
            transform: Optional transformation applied to each data sample.
        """
        super().__init__(**kwargs)
        # atom_properties and bond_properties are given as lists containing class_paths
        if properties is not None:
            properties = [resolve_property(prop) for prop in properties]
            properties = self._sort_properties(properties)
        else:
            properties = []
        self.properties = properties
        assert isinstance(self.properties, list) and all(
            isinstance(p, MolecularProperty) for p in self.properties
        )
        self.transform = transform

    def _sort_properties(
        self, properties: list[MolecularProperty]
    ) -> list[MolecularProperty]:
        return sorted(properties, key=lambda prop: self.get_property_path(prop))

    def _setup_properties(self) -> None:
        """
        Process and cache molecular properties to disk.

        Returns:
            None
        """
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
        def enc_if_not_none(encode, value):
            return (
                [encode(v) for v in value]
                if value is not None and len(value) > 0
                else None
            )

        for property in self.properties:
            if not os.path.isfile(self.get_property_path(property)):
                rank_zero_info(f"Processing property {property.name}")
                # read all property values first, then encode
                rank_zero_info(f"\tReading property values of {property.name}...")
                property_values = [
                    self.reader.read_property(feat, property)
                    for feat in tqdm.tqdm(features)
                ]
                rank_zero_info(f"\tEncoding property values of {property.name}...")
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
    def processed_properties_dir(self) -> str:
        return os.path.join(self.processed_dir, "properties")

    def get_property_path(self, property: MolecularProperty) -> str:
        """
        Construct the cache path for a given molecular property.

        Args:
            property: Instance of a MolecularProperty.

        Returns:
            Path to the cached property file.
        """
        return os.path.join(
            self.processed_properties_dir,
            f"{property.name}_{property.encoder.name}.pt",
        )

    def _after_setup(self, **kwargs) -> None:
        """
        Finalize setup after ensuring properties are processed.

        Args:
            **kwargs: Additional keyword arguments passed to superclass.

        Returns:
            None
        """
        self._setup_properties()
        super()._after_setup(**kwargs)


class GraphPropertiesMixIn(DataPropertiesSetter, ABC):
    def __init__(
        self,
        properties=None,
        transform=None,
        pad_node_features: int = None,
        pad_edge_features: int = None,
        distribution: str = "normal",
        **kwargs,
    ):
        super().__init__(properties, transform, **kwargs)
        self.pad_edge_features = int(pad_edge_features) if pad_edge_features else None
        self.pad_node_features = int(pad_node_features) if pad_node_features else None
        if self.pad_node_features or self.pad_edge_features:
            assert (
                distribution is not None
                and distribution in RandomFeatureInitializationReader.DISTRIBUTIONS
            ), "When using padding for features, a valid distribution must be specified."
            self.distribution = distribution
            if self.pad_node_features:
                print(
                    f"[Info] Node-level features will be padded with random"
                    f"{self.pad_node_features} values from {self.distribution} distribution."
                )
            if self.pad_edge_features:
                print(
                    f"[Info] Edge-level features will be padded with random"
                    f"{self.pad_edge_features} values from {self.distribution} distribution."
                )

        if self.properties:
            print(
                f"Data module uses these properties (ordered): {', '.join([str(p) for p in self.properties])}"
            )

    def _merge_props_into_base(self, row: pd.Series) -> GeomData:
        """
        Merge encoded molecular properties into the GeomData object.

        Args:
            row: A dictionary containing 'features' and encoded properties.

        Returns:
            A GeomData object with merged features.
        """
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
                # Concat/Duplicate properties values for undirected graph as `edge_index` has first src to tgt edges, then tgt to src edges
                edge_attr = torch.cat(
                    [edge_attr, torch.cat([property_values, property_values], dim=0)],
                    dim=1,
                )
            elif isinstance(property, MoleculeProperty):
                molecule_attr = torch.cat([molecule_attr, property_values], dim=1)
            else:
                raise TypeError(f"Unsupported property type: {type(property).__name__}")

        if self.pad_node_features:
            padding_values = torch.empty((x.shape[0], self.pad_node_features))
            RandomFeatureInitializationReader.random_gni(
                padding_values, self.distribution
            )
            x = torch.cat([x, padding_values], dim=1)

        if self.pad_edge_features:
            padding_values = torch.empty((edge_attr.shape[0], self.pad_edge_features))
            RandomFeatureInitializationReader.random_gni(
                padding_values, self.distribution
            )
            edge_attr = torch.cat([edge_attr, padding_values], dim=1)

        return GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
            molecule_attr=molecule_attr,
        )

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> list[dict]:
        """
        Load dataset and merge cached properties into base features.

        Args:
            filename: The path to the file to load.

        Returns:
            List of data entries, each a dictionary.
        """
        base_data = super().load_processed_data(kind, filename)
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

        # -------------------------- Count total node properties
        n_node_properties = sum(
            p.encoder.get_encoding_length()
            for p in self.properties
            if isinstance(p, AtomProperty)
        )

        in_channels_str = ""
        if self.pad_node_features:
            n_node_properties += self.pad_node_features
            in_channels_str += f" (with {self.pad_node_features} padded random values from {self.distribution} distribution)"

        in_channels_str = f"in_channels: {n_node_properties}" + in_channels_str

        # -------------------------- Count total edge properties
        n_edge_properties = sum(
            p.encoder.get_encoding_length()
            for p in self.properties
            if isinstance(p, BondProperty)
        )
        edge_dim_str = ""
        if self.pad_edge_features:
            n_edge_properties += self.pad_edge_features
            edge_dim_str += f" (with {self.pad_edge_features} padded random values from {self.distribution} distribution)"

        edge_dim_str = f"edge_dim: {n_edge_properties}" + edge_dim_str

        rank_zero_info(
            f"Finished loading dataset from properties.\nEncoding lengths: {prop_lengths}\n"
            f"Use following values for given parameters for model configuration: \n\t"
            f"{in_channels_str} \n\t"
            f"{edge_dim_str} \n\t"
            f"n_molecule_properties: {sum(p.encoder.get_encoding_length() for p in self.properties if isinstance(p, MoleculeProperty))}"
        )

        return base_df[base_data[0].keys()].to_dict("records")


class GraphPropAsPerNodeType(DataPropertiesSetter, ABC):
    def __init__(self, properties=None, transform=None, **kwargs):
        super().__init__(properties, transform, **kwargs)
        # Sort properties so that AllNodeTypeProperty instances come first, rest of the properties order remain same
        first = self._sort_properties(
            [prop for prop in self.properties if isinstance(prop, AllNodeTypeProperty)]
        )
        rest = self._sort_properties(
            [
                prop
                for prop in self.properties
                if not isinstance(prop, AllNodeTypeProperty)
            ]
        )
        self.properties = first + rest
        print(
            "Properties are sorted so that `AllNodeTypeProperty` properties are first in sequence and rest of the order remains same\n",
            f"Data module uses these properties (ordered): {', '.join([str(p) for p in self.properties])}",
        )

    def load_processed_data(
        self, kind: Optional[str] = None, filename: Optional[str] = None
    ) -> list[dict]:
        """
        Load dataset and merge cached properties into base features.

        Args:
            filename: The path to the file to load.

        Returns:
            List of data entries, each a dictionary.
        """
        base_data = super().load_processed_data(kind, filename)
        base_df = pd.DataFrame(base_data)
        props_categories = {
            "AllNodeTypeProperties": [],
            "FGNodeTypeProperties": [],
            "AtomNodeTypeProperties": [],
            "GraphNodeTypeProperties": [],
            "BondProperties": [],
        }
        n_atom_node_properties, n_fg_node_properties = 0, 0
        n_bond_properties, n_graph_node_properties = 0, 0
        prop_lengths = []
        for prop in self.properties:
            prop_length = prop.encoder.get_encoding_length()
            prop_name = prop.name
            prop_lengths.append((prop_name, prop_length))
            if isinstance(prop, AllNodeTypeProperty):
                n_atom_node_properties += prop_length
                n_fg_node_properties += prop_length
                n_graph_node_properties += prop_length
                props_categories["AllNodeTypeProperties"].append(prop_name)
            elif isinstance(prop, FGNodeTypeProperty):
                n_fg_node_properties += prop_length
                props_categories["FGNodeTypeProperties"].append(prop_name)
            elif isinstance(prop, AtomNodeTypeProperty):
                n_atom_node_properties += prop_length
                props_categories["AtomNodeTypeProperties"].append(prop_name)
            elif isinstance(prop, BondProperty):
                n_bond_properties += prop_length
                props_categories["BondProperties"].append(prop_name)
            elif isinstance(prop, MoleculeProperty):
                # molecule props will be used as graph node props
                n_graph_node_properties += prop_length
                props_categories["GraphNodeTypeProperties"].append(prop_name)
            else:
                raise TypeError(f"Unsupported property type: {type(prop).__name__}")

        n_node_properties = max(
            n_atom_node_properties, n_fg_node_properties, n_graph_node_properties
        )
        rank_zero_info(
            f"\nFinished loading dataset from properties.\nEncoding lengths: {prop_lengths}\n\n"
            f"Properties Categories:\n{pformat(props_categories)}\n\n"
            f"n_atom_node_properties: {n_atom_node_properties}, "
            f"n_fg_node_properties: {n_fg_node_properties}, "
            f"n_bond_properties: {n_bond_properties}, "
            f"n_graph_node_properties: {n_graph_node_properties}\n\n"
            f"Use following values for given parameters for model configuration: \n\t"
            f"in_channels: {n_node_properties}, edge_dim: {n_bond_properties}, n_molecule_properties: 0\n"
        )

        for property in self.properties:
            rank_zero_info(f"Loading property {property.name}...")
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
            lambda row: self._merge_props_into_base(
                row,
                max_len_node_properties=n_node_properties,
            ),
            axis=1,
        )

        # apply transformation, e.g. masking for pretraining task
        if self.transform is not None:
            base_df["features"] = base_df["features"].apply(self.transform)

        return base_df[base_data[0].keys()].to_dict("records")

    def _merge_props_into_base(
        self, row: pd.Series, max_len_node_properties: int
    ) -> GeomData:
        """
        Merge encoded molecular properties into the GeomData object.

        Args:
            row: A dictionary containing 'features' and encoded properties.

        Returns:
            A GeomData object with merged features.
        """
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData)

        is_atom_node = geom_data.is_atom_node
        assert is_atom_node is not None, "`is_atom_node` must be set in the geom_data"
        is_graph_node = geom_data.is_graph_node
        assert is_graph_node is not None, "`is_graph_node` must be set in the geom_data"

        is_fg_node = ~is_atom_node & ~is_graph_node
        num_nodes = geom_data.x.size(0)
        edge_attr = geom_data.edge_attr

        # Initialize node feature matrix
        assert (
            max_len_node_properties is not None
        ), "Maximum len of node properties should not be None"
        x = torch.zeros((num_nodes, max_len_node_properties))

        # Track column offsets for each node type
        atom_offset, fg_offset, graph_offset = 0, 0, 0

        for property in self.properties:
            property_values = row[f"{property.name}"].to(dtype=torch.float32)
            if isinstance(property_values, torch.Tensor):
                if len(property_values.size()) == 0:
                    property_values = property_values.unsqueeze(0)
                if len(property_values.size()) == 1:
                    property_values = property_values.unsqueeze(1)
            else:
                property_values = torch.zeros(
                    (0, property.encoder.get_encoding_length())
                )

            enc_len = property_values.shape[1]
            # -------------- Node properties ---------------
            if isinstance(property, AllNodeTypeProperty):
                x[:, atom_offset : atom_offset + enc_len] = property_values
                atom_offset += enc_len
                fg_offset += enc_len
                graph_offset += enc_len

            elif isinstance(property, AtomNodeTypeProperty):
                x[is_atom_node, atom_offset : atom_offset + enc_len] = property_values[
                    is_atom_node
                ]
                atom_offset += enc_len

            elif isinstance(property, FGNodeTypeProperty):
                x[is_fg_node, fg_offset : fg_offset + enc_len] = property_values[
                    is_fg_node
                ]
                fg_offset += enc_len

            elif isinstance(property, MoleculeProperty):
                x[is_graph_node, graph_offset : graph_offset + enc_len] = (
                    property_values
                )
                graph_offset += enc_len

            # ------------- Bond Properties --------------
            elif isinstance(property, BondProperty):
                # Concat/Duplicate properties values for undirected graph as `edge_index` has first src to tgt edges, then tgt to src edges
                edge_attr = torch.cat(
                    [edge_attr, torch.cat([property_values, property_values], dim=0)],
                    dim=1,
                )
            else:
                raise TypeError(f"Unsupported property type: {type(property).__name__}")

            total_used_columns = max(atom_offset, fg_offset, graph_offset)
            assert (
                total_used_columns <= max_len_node_properties
            ), f"Used {total_used_columns} columns, but max allowed is {max_len_node_properties}"

        return GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
            molecule_attr=torch.empty((1, 0)),  # empty as not used for this class
            is_atom_node=is_atom_node,
            is_fg_node=is_fg_node,
            is_graph_node=is_graph_node,
        )


class ChEBI50_StaticGNI(DataPropertiesSetter, ChEBIOver50):
    READER = RandomFeatureInitializationReader

    def _setup_properties(self): ...

    def load_processed_data_from_file(self, filename):
        base_data = super().load_processed_data_from_file(filename)
        base_df = pd.DataFrame(base_data)

        rank_zero_info(
            f"Use following values for given parameters for model configuration: \n\t"
            f"in_channels: {self.reader.num_node_properties} , "
            f"edge_dim: {self.reader.num_bond_properties}, "
            f"n_molecule_properties: {self.reader.num_molecule_properties}"
        )
        return base_df[base_data[0].keys()].to_dict("records")


class ChEBI50GraphProperties(GraphPropertiesMixIn, ChEBIOver50):
    """ChEBIOver50 dataset with molecular property encodings."""

    pass


class ChEBI100GraphProperties(GraphPropertiesMixIn, ChEBIOver100):
    """ChEBIOver100 dataset with molecular property encodings."""

    pass


class ChEBI50GraphPropertiesPartial(ChEBI50GraphProperties, ChEBIOverXPartial):
    """Partial version of ChEBIOver50 with molecular properties."""

    pass


class AugGraphPropMixIn_NoGraphNode(GraphPropertiesMixIn, ABC):
    """Mixin for augmented graph data without additional graph nodes."""

    READER = None

    def _merge_props_into_base(self, row: pd.Series) -> GeomData:
        data = super()._merge_props_into_base(row)
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData) and isinstance(data, GeomData)

        is_atom_node = geom_data.is_atom_node
        assert is_atom_node is not None, "is_atom_node must be set in the geom_data"
        data.is_atom_node = is_atom_node
        return data


class AugGraphPropMixIn_WithGraphNode(AugGraphPropMixIn_NoGraphNode, ABC):
    """Mixin for augmented graph data with graph-level nodes."""

    READER = None

    def _merge_props_into_base(self, row: pd.Series) -> GeomData:
        data = super()._merge_props_into_base(row)
        return self._add_graph_node_mask(data, row)

    def _add_graph_node_mask(self, data: GeomData, row: pd.Series) -> GeomData:
        """
        Add a graph node mask to the GeomData object.

        Args:
            data: A GeomData object with features.
            row: A dictionary containing 'features' and other metadata.

        Returns:
            Modified GeomData with graph node mask added.
        """
        geom_data = row["features"]
        assert isinstance(geom_data, GeomData) and isinstance(data, GeomData)
        is_graph_node = geom_data.is_graph_node
        assert is_graph_node is not None, "is_graph_node must be set in the geom_data"
        data.is_graph_node = is_graph_node
        return data


class ChEBI50_WFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with with FG nodes and FG edges and graph node."""

    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI50_GN_WithAllNodes_FG_WithAtoms_FGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_FGE


class ChEBI50_GN_WithAllNodes_FG_WithAtoms_NoFGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all nodes (fg + atoms).
    """

    READER = GN_WithAllNodes_FG_WithAtoms_NoFGE


class ChEBI50_GN_WithAtoms_FG_WithAtoms_FGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) with functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_FGE


class ChEBI50_GN_WithAtoms_FG_WithAtoms_NoFGE(
    AugGraphPropMixIn_WithGraphNode, ChEBIOver50
):
    """
    ChEBIOver50 with FG nodes (connected to their respective atom nodes) without functional group
    edges, and adds a graph-level node connected to all atom nodes.
    """

    READER = GN_WithAtoms_FG_WithAtoms_NoFGE


class ChEBI50_NFGE_WGN_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with FG nodes but without FG edges, with graph node."""

    READER = AtomFGReader_NoFGEdges_WithGraphNode


class ChEBI50_WFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, ChEBIOver50):
    """ChEBIOver50 with FG nodes and FG edges, no graph node."""

    READER = AtomFGReader_WithFGEdges_NoGraphNode


class ChEBI50_NFGE_NGN_GraphProp(AugGraphPropMixIn_NoGraphNode, ChEBIOver50):
    """ChEBIOver50 with FG nodes but without FG edges or graph node."""

    READER = AtomsFGReader_NoFGEdges_NoGraphNode


class ChEBI50_Atom_WGNOnly_GraphProp(AugGraphPropMixIn_WithGraphNode, ChEBIOver50):
    """ChEBIOver50 with atom-level nodes and graph node only."""

    READER = AtomReader_WithGraphNodeOnly


class ChEBI50_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOver50):
    READER = AtomFGReader_WithFGEdges_WithGraphNode


class ChEBI100_WFGE_WGN_AsPerNodeType(GraphPropAsPerNodeType, ChEBIOver100):
    READER = AtomFGReader_WithFGEdges_WithGraphNode
