
class GATOnSWJ(Experiment):
    MODEL = graph.JCIGraphAttentionNet

    @classmethod
    def identifier(cls) -> str:
        return "GAT+JCIExt"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            in_length=50,
            hidden_length=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.JCIGraphData(batch_size)


class GATOnTox21(Experiment):
    MODEL = graph.JCIGraphAttentionNet

    @classmethod
    def identifier(cls) -> str:
        return "GAT+Tox21"

    def model_kwargs(self, *args) -> Dict:
        return dict(
            in_length=50,
            hidden_length=100,
        )

    def build_dataset(self, batch_size) -> datasets.XYBaseDataModule:
        return datasets.Tox21Graph(batch_size)
