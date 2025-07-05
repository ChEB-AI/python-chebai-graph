import torch
from chebai.preprocessing.collate import RaggedCollator
from torch_geometric.data import Data as GeomData
from torch_geometric.data.collate import collate as graph_collate

from chebai_graph.preprocessing.structures import XYGraphData


class GraphCollator(RaggedCollator):
    """Collates a batch of molecular graph data with label handling and edge consistency."""

    def __call__(self, data):
        loss_kwargs: dict = {}

        # Unpack labels and optional identifiers
        y, idents = zip(*((d["labels"], d.get("ident")) for d in data))

        # Replace labels with `y` inside graph features and collect them
        merged_data = []
        for row in data:
            row["features"].y = row["labels"]
            merged_data.append(row["features"])

        # Add empty edge_attr for graphs with no edges to prevent PyG errors
        for mdata in merged_data:
            for store in mdata.stores:
                if "edge_attr" not in store:
                    store["edge_attr"] = torch.tensor([])

        # Ensure all attributes are float tensors to prevent torch.cat dtype issues
        for attr in merged_data[0].keys():
            for data in merged_data:
                for store in data.stores:
                    # Im not sure why the following conversion is needed, but it solves this error:
                    # packages/torch_geometric/data/collate.py", line 177, in _collate
                    #     value = torch.cat(values, dim=cat_dim or 0, out=out)
                    # RuntimeError: torch.cat(): input types can't be cast to the desired output type Long
                    if isinstance(store[attr], torch.Tensor):
                        store[attr] = store[attr].to(dtype=torch.float32)
                    else:
                        store[attr] = torch.tensor(store[attr], dtype=torch.float32)

        # Use PyG's batch collate for graph data
        x = graph_collate(
            GeomData,
            merged_data,
            follow_batch=["x", "edge_attr", "edge_index", "label"],
        )

        # Handle various combinations of missing or available labels
        if any(x is not None for x in y):
            # If any label is not None: (None, None, `1`, None)
            if any(x is None for x in y):
                # If any label is None: (`None`, `None`, 1, `None`)
                non_null_labels = [i for i, r in enumerate(y) if r is not None]
                y = self.process_label_rows(
                    tuple(ye for i, ye in enumerate(y) if i in non_null_labels)
                )
                loss_kwargs["non_null_labels"] = non_null_labels
            else:
                # If all labels are not None: (`0`, `2`, `1`, `3`)
                y = self.process_label_rows(y)
        else:
            # If all labels are None: e.g., (None, None, None, None)
            y = None
            loss_kwargs["non_null_labels"] = []

        # Set node features (x) to long dtype (e.g., for categorical features)
        x[0].x = x[0].x.to(dtype=torch.int64)
        # x is a Tuple[BaseData, Mapping, Mapping]

        return XYGraphData(
            x,
            y,
            idents=idents,
            model_kwargs={},
            loss_kwargs=loss_kwargs,
        )
