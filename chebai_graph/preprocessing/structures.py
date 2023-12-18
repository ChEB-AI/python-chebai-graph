from chebai.preprocessing.structures import XYData


class XYGraphData(XYData):
    def __len__(self):
        return len(self.y)

    def to_x(self, device):
        if isinstance(self.x, tuple):
            res = []
            for elem in self.x:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        elem[k] = v.to(device) if v is not None else None
                else:
                    elem = elem.to(device)
                res.append(elem)
            return tuple(res)
        return super(self, device)
