import torch


class MaskPretrainingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target, **loss_kwargs):
        print(
            f"Called loss function with input: {input} (type: {type(input)}, target: {target}"
        )
        for i in input:
            print(f"Input i: type: {type(i)}")
            if type(i) == torch.Tensor:
                print(f"Input i: shape: {i.shape}")
            print(f"Input i: {i}")
        return torch.tensor(0)
