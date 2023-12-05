from torch_geometric.data import Data

class PrePairData(Data):
    def __init__(self, l=None, r=None, label=None):
        super(PrePairData, self).__init__()
        self.l = l
        self.r = r
        self.label = label


class PairData(Data):
    def __init__(self, ppd: PrePairData, graph):
        super(PairData, self).__init__()

        s = graph.nodes[ppd.l]["enc"]
        self.edge_index_s = s.edge_index
        self.x_s = s.x

        t = graph.nodes[ppd.r]["enc"]
        self.edge_index_t = t.edge_index
        self.x_t = t.x

        self.label = ppd.label

    def __inc__(self, key, value):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)