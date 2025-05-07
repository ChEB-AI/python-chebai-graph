import random

import torch


# class taken from Hu, 2020: https://arxiv.org/pdf/1905.12265, modified
# acts as a transformation for input data, masking some atoms and edges
class MaskAtom:
    def __init__(
        self, mask_rate=0.15, mask_edge=True, n_bond_properties=7, random_seed=42
    ):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.n_bond_properties = n_bond_properties
        self.random_seed = random_seed

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices is None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            random.seed(self.random_seed)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # within atoms, only mask some properties
        mask_n_properties = int(data.x.size()[1] * self.mask_rate + 1)
        random.seed(self.random_seed)
        masked_property_indices = [
            random.sample(range(data.x.size()[1]), mask_n_properties)
            for _ in masked_atom_indices
        ]

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx, property_idxs in zip(
            masked_atom_indices, masked_property_indices
        ):
            mask_node_labels_list.append(data.x[atom_idx, property_idxs].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)
        data.masked_property_indices = torch.tensor(masked_property_indices)

        # modify the original node feature of the masked node
        for atom_idx, property_idxs in zip(
            masked_atom_indices, masked_property_indices
        ):
            data.x[atom_idx, property_idxs] = torch.zeros(mask_n_properties)

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in {u, v} and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.zeros(data.edge_attr.size()[1])

                data.connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, self.n_bond_properties)).to(
                    torch.int64
                )
                data.connected_edge_indices = torch.tensor(connected_edge_indices).to(
                    torch.int64
                )

        return data
