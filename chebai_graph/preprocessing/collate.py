from typing import Dict

import torch
from chebai.preprocessing.collate import RaggedCollator
from torch_geometric.data import Data as GeomData
from torch_geometric.data.collate import collate as graph_collate

from chebai_graph.preprocessing.structures import XYGraphData


class GraphCollator(RaggedCollator):
    def __call__(self, data):
        loss_kwargs: Dict = dict()

        y, idents = zip(*((d["labels"], d.get("ident")) for d in data))
        merged_data = []
        for row in data:
            row["features"].y = row["labels"]
            merged_data.append(row["features"])
        # add empty edge_attr to avoid problems during collate (only relevant for molecules without edges)
        for mdata in merged_data:
            for i, store in enumerate(mdata.stores):
                if "edge_attr" not in store:
                    store["edge_attr"] = torch.tensor([])
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

        x = graph_collate(
            GeomData,
            merged_data,
            follow_batch=["x", "edge_attr", "edge_index", "label"],
        )
        if any(x is not None for x in y):
            if any(x is None for x in y):
                non_null_labels = [i for i, r in enumerate(y) if r is not None]
                y = self.process_label_rows(
                    tuple(ye for i, ye in enumerate(y) if i in non_null_labels)
                )
                loss_kwargs["non_null_labels"] = non_null_labels
            else:
                y = self.process_label_rows(y)
        else:
            y = None
            loss_kwargs["non_null_labels"] = []

        x[0].x = x[0].x.to(dtype=torch.int64)
        # x is a Tuple[BaseData, Mapping, Mapping]
        return XYGraphData(
            x,
            y,
            idents=idents,
            model_kwargs={},
            loss_kwargs=loss_kwargs,
        )
