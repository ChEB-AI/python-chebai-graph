from typing import Any

import torch
from chebai.preprocessing.structures import XYData
from torch_geometric.data import Data as GraphData

from chebai_graph.models.architectures.base import GraphBaseNet
from chebai_graph.models.architectures.gat import GATGraphConvNetBase


class GATMSPretraining(GraphBaseNet):
    def __init__(
        self,
        config: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn = GATGraphConvNetBase(config=config)

    def _process_batch(self, batch: XYData, batch_idx: int) -> dict[str, Any]:
        """
        Processes the batch data.

        Args:
            batch (XYData): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Dict[str, Any]: Processed batch data.
        """
        return dict(
            features=batch.x,
            labels=self._process_labels_in_batch(batch),
            model_kwargs=batch.additional_fields["model_kwargs"],
            loss_kwargs=batch.additional_fields["loss_kwargs"],
            idents=batch.additional_fields["idents"],
            custom_metadata=batch.additional_fields["custom_metadata"],
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through GNN, pooling and linear layers.

        Args:
            batch (dict): Input batch with graph features.

        Returns:
            torch.Tensor: Predicted output.
        """
        graph_data = batch["features"][0]
        graph_data.to(self.device)
        assert isinstance(graph_data, GraphData)
        return self.gnn(batch)

    def _process_for_loss(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        data: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Processes the data for loss computation.

        Args:
            model_output (torch.Tensor): The model output.
            labels (torch.Tensor): The true labels.
            data (dict[str, Any]): The processed batch data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict[str, Any]]: Model output, labels, and data.
        """
        data["valid_label_mask"] = data.get("loss_kwargs", dict()).get(
            "valid_label_mask", None
        )
        return model_output, labels, data
