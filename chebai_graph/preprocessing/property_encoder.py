import abc
import os
import torch
from typing import Optional


class PropertyEncoder(abc.ABC):
    def __init__(self, property, **kwargs):
        self.property = property
        self._encoding_length = 1

    @property
    def name(self):
        return ""

    def get_encoding_length(self) -> int:
        return self._encoding_length

    def set_encoding_length(self, encoding_length: int) -> None:
        self._encoding_length = encoding_length

    def encode(self, value):
        raise NotImplementedError

    def on_start(self, **kwargs):
        pass

    def on_finish(self):
        return


class IndexEncoder(PropertyEncoder):
    """Enocdes property values as indices. For that purpose, compiles a dynamic list of different values that have
    occured. Stores this list in a file for later reference."""

    def __init__(self, property, indices_dir=None, **kwargs):
        super().__init__(property, **kwargs)
        if indices_dir is None:
            indices_dir = os.path.dirname(__file__)
        self.dirname = indices_dir
        # load already existing cache
        with open(self.index_path, "r") as pk:
            self.cache = [x.strip() for x in pk]
        self.index_length_start = len(self.cache)
        self.offset = 0

    @property
    def name(self):
        return "index"

    @property
    def index_path(self):
        """Get path to store indices of property values, create file if it does not exist yet"""
        index_path = os.path.join(
            self.dirname, "bin", self.property.name, "indices.txt"
        )
        os.makedirs(
            os.path.join(self.dirname, "bin", self.property.name), exist_ok=True
        )
        if not os.path.exists(index_path):
            with open(index_path, "x"):
                pass
        return index_path

    def on_finish(self):
        """Save cache"""
        with open(self.index_path, "w") as pk:
            new_length = len(self.cache) - self.index_length_start
            pk.writelines([f"{c}\n" for c in self.cache])
            print(
                f"saved index of property {self.property.name} to {self.index_path}, "
                f"index length: {len(self.cache)} (new: {new_length})"
            )

    def encode(self, token):
        """Returns a unique number for each token, automatically adds new tokens to the cache."""
        if not str(token) in self.cache:
            self.cache.append(str(token))
        return torch.tensor([self.cache.index(str(token)) + self.offset])


class OneHotEncoder(IndexEncoder):
    """Returns one-hot encoding of the value (position in one-hot vector is defined by index)."""

    def __init__(self, property, n_labels: Optional[int] = None, **kwargs):
        super().__init__(property, **kwargs)
        self._encoding_length = n_labels

    def get_encoding_length(self) -> int:
        return self._encoding_length or len(self.cache)

    @property
    def name(self):
        return f"one_hot"

    def on_start(self, property_values):
        """To get correct number of classes during encoding, cache unique tokens beforehand"""
        unique_tokens = list(
            dict.fromkeys(
                [
                    v
                    for vs in property_values
                    if vs is not None
                    for v in vs
                    if v is not None
                ]
            )
        )
        self.tokens_dict = {}
        for token in unique_tokens:
            self.tokens_dict[token] = super().encode(token)

    def encode(self, token):
        return torch.nn.functional.one_hot(
            self.tokens_dict[token], num_classes=self.get_encoding_length()
        )


class AsIsEncoder(PropertyEncoder):
    """Returns the input value as it is, useful e.g. for float values."""

    @property
    def name(self):
        return "asis"

    def encode(self, token):
        return torch.tensor([token])


class BoolEncoder(PropertyEncoder):
    @property
    def name(self):
        return "bool"

    def encode(self, token: bool):
        return torch.tensor([1 if token else 0])
