import torch
from torch_geometric.data import Data as GeomData
from torch_geometric.data.collate import collate as graph_collate
from chebai_graph.preprocessing.structures import XYGraphData

from chebai.preprocessing.collate import RaggedCollator


class GraphCollator(RaggedCollator):
    def __call__(self, data):
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
        y = self.process_label_rows(y)
        x[0].x = x[0].x.to(dtype=torch.int64)
        # x is a Tuple[BaseData, Mapping, Mapping]
        return XYGraphData(
            x,
            y,
            idents=idents,
            model_kwargs={},
            loss_kwargs={},
        )
