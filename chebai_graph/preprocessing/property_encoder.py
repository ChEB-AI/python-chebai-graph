import abc
import inspect
import os
import sys
from itertools import islice
from typing import Optional

import torch


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
        return value

    def on_start(self, **kwargs):
        pass

    def on_finish(self):
        return


class IndexEncoder(PropertyEncoder):
    """Encodes property values as indices. For that purpose, compiles a dynamic list of different values that have
    occurred. Stores this list in a file for later reference."""

    def __init__(self, property, indices_dir=None, **kwargs):
        super().__init__(property, **kwargs)
        if indices_dir is None:
            indices_dir = os.path.dirname(inspect.getfile(self.__class__))
        self.dirname = indices_dir
        # load already existing cache
        with open(self.index_path, "r") as pk:
            self.cache: dict[str, int] = {
                token.strip(): idx for idx, token in enumerate(pk)
            }
        self.index_length_start = len(self.cache)
        self._unk_token_idx = 0
        self._count_for_unk_token = 0
        self.offset = 1

    @property
    def name(self):
        return "index"

    @property
    def index_path(self):
        """Get path to store indices of property values, create file if it does not exist yet"""
        index_path = os.path.join(
            self.dirname, "bin", self.property.name, f"indices_{self.name}.txt"
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
        total_tokens = len(self.cache)
        if total_tokens > self.index_length_start:
            print("New tokens added to the cache, Saving them to index token file.....")

            assert sys.version_info >= (
                3,
                7,
            ), "This code requires Python 3.7 or higher."
            # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
            # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
            # https://mail.python.org/pipermail/python-dev/2017-December/151283.html
            new_tokens = list(islice(self.cache, self.index_length_start, total_tokens))

            with open(self.index_path, "a") as pk:
                pk.writelines([f"{c}\n" for c in new_tokens])
                print(
                    f"New {len(new_tokens)} tokens append to index of property {self.property.name} to {self.index_path}..."
                )
                print(
                    f"Now, the total length of the index of property {self.property.name} is {total_tokens}"
                )

        if self._count_for_unk_token > 0:
            print(
                f"{self.__class__.__name__} Encountered {self._count_for_unk_token} unknown tokens"
            )

    def encode(self, token):
        """Returns a unique number for each token, automatically adds new tokens to the cache."""
        if token is None:
            self._count_for_unk_token += 1
            return torch.tensor([self._unk_token_idx])

        if str(token) not in self.cache:
            self.cache[(str(token))] = len(self.cache)
        return torch.tensor([self.cache[str(token)] + self.offset])


class OneHotEncoder(IndexEncoder):
    """Returns one-hot encoding of the value (position in one-hot vector is defined by index)."""

    def __init__(self, property, n_labels: Optional[int] = None, **kwargs):
        super().__init__(property, **kwargs)
        self._encoding_length = n_labels
        # To undo any offset set by index encoder as its not relevant for one-hot-encoder (no offset needed for some unknown/reserved token)
        # Also, `torch.nn.functional.one_hot` that class values must be smaller than num_classes.
        self.offset = 0

    def get_encoding_length(self) -> int:
        return self._encoding_length or len(self.cache)

    @property
    def name(self):
        return "one_hot"

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
        if token not in self.tokens_dict:
            self._count_for_unk_token += 1
            return torch.zeros(1, self.get_encoding_length(), dtype=torch.int64)

        return torch.nn.functional.one_hot(
            self.tokens_dict[token], num_classes=self.get_encoding_length()
        )


class AsIsEncoder(PropertyEncoder):
    """Returns the input value as it is, useful e.g. for float values."""

    @property
    def name(self):
        return "asis"

    def encode(self, token):
        if token is None:
            return torch.tensor([0])
        return torch.tensor([token])


class BoolEncoder(PropertyEncoder):
    @property
    def name(self):
        return "bool"

    def encode(self, token: bool):
        return torch.tensor([1 if token else 0])
