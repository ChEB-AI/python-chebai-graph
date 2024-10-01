import torch


class MaskPretrainingLoss(torch.nn.Module):
    # Mask atoms and edges, try to predict them (see Hu et al., 2020: Strategies for Pre-training Graph Neural Networks)
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target, **loss_kwargs):
        atom_preds, bond_preds = input
        atom_targets, bond_targets = target
        atom_loss = self.ce(atom_preds, atom_targets)
        bond_loss = self.ce(bond_preds, bond_targets)
        return atom_loss + bond_loss
