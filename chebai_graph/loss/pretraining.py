import torch


class MaskPretrainingLoss(torch.nn.Module):
    # Mask atoms and edges, try to predict them (see Hu et al., 2020: Strategies for Pre-training Graph Neural Networks)
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.functional.binary_cross_entropy_with_logits

    def forward(self, input, target, **loss_kwargs):
        if isinstance(input, tuple):
            atom_preds, bond_preds = input
            atom_targets, bond_targets = target
            try:
                bond_loss = self.ce(bond_preds, bond_targets)
            except RuntimeError as e:
                print(f"Failed to compute bond loss: {e}")
                print(f"Input: preds: {bond_preds.shape}, labels: {bond_targets.shape}")
                bond_loss = 0
        else:
            atom_preds = input
            atom_targets = target
            bond_loss = 0
        try:
            atom_loss = self.ce(atom_preds, atom_targets)
        except RuntimeError as e:
            print(f"Failed to compute atom loss: {e}")
            print(f"Input: preds: {atom_preds.shape}, labels: {atom_targets.shape}")
            atom_loss = 0

        return atom_loss + bond_loss
