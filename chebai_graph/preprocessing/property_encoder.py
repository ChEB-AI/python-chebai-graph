import abc
import inspect
import os
import sys
from itertools import islice

import torch


class PropertyEncoder(abc.ABC):
    """
    Abstract base class for encoding property values.

    Args:
        property: The property object associated with this encoder.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, property, eval=False, **kwargs) -> None:
        self.property = property
        self._encoding_length: int = 1
        self.eval = eval  # if True, do not update cache (for index encoder)

    @property
    def name(self) -> str:
        """Name of the encoder."""
        return ""

    def get_encoding_length(self) -> int:
        """Return the length of the encoding vector."""
        return self._encoding_length

    def set_encoding_length(self, encoding_length: int) -> None:
        """Set the length of the encoding vector."""
        self._encoding_length = encoding_length

    def encode(self, value) -> torch.Tensor:
        """
        Encode the given value.

        Args:
            value: The value to encode.

        Returns:
            Encoded tensor.
        """
        return value

    def on_start(self, **kwargs) -> None:
        """Hook called at the start of encoding process."""
        pass

    def on_finish(self) -> None:
        """Hook called at the end of encoding process."""
        return


class IndexEncoder(PropertyEncoder):
    """
    Encodes property values as indices. For that purpose, compiles a dynamic list of different values that have
    occurred. Stores this list in a file for later reference.

    Args:
        property: The property object.
        indices_dir: Optional directory to store index files.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, property, indices_dir: str | None = None, **kwargs) -> None:
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
    def name(self) -> str:
        """Name of this encoder."""
        return "index"

    @property
    def index_path(self) -> str:
        """Get path to store indices of property values, create file if it does not exist yet

        Returns:
            Path to index file.
        """
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

    def on_finish(self) -> None:
        """
        Save cache

        Saves new tokens added to the cache to the index file and logs count of unknown tokens.
        """
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

    def encode(self, token: str | None) -> torch.Tensor:
        """
        Returns a unique number for each token, automatically adds new tokens to the cache.

        Args:
            token: The token to encode.

        Returns:
            A tensor containing the encoded index.
        """
        if token is None:
            self._count_for_unk_token += 1
            return torch.tensor([self._unk_token_idx])

        if self.eval and str(token) not in self.cache:
            self._count_for_unk_token += 1
            return torch.tensor([self._unk_token_idx])

        if str(token) not in self.cache:
            self.cache[str(token)] = len(self.cache)
        return torch.tensor([self.cache[str(token)] + self.offset])


class OneHotEncoder(IndexEncoder):
    """
    Returns one-hot encoding of the value (position in one-hot vector is defined by index).

    Args:
        property: The property object.
        n_labels: Optional number of labels for encoding.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, property, n_labels: int | None = None, **kwargs) -> None:
        super().__init__(property, **kwargs)
        self._encoding_length = n_labels
        # To undo any offset set by index encoder as its not relevant for one-hot-encoder (no offset needed for some unknown/reserved token)
        # Also, `torch.nn.functional.one_hot` that class values must be smaller than num_classes.
        self.offset = 0

    def get_encoding_length(self) -> int:
        """Return the number of classes for one-hot encoding."""
        return self._encoding_length or len(self.cache)

    @property
    def name(self) -> str:
        """Name of this encoder."""
        return "one_hot"

    def on_start(self, property_values: list[list[str | None]]) -> None:
        """
        To get correct number of classes during encoding, cache unique tokens beforehand

        Args:
            property_values: List of property value sequences.
        """
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
        self.tokens_dict: dict[str, torch.Tensor] = {}
        for token in unique_tokens:
            self.tokens_dict[token] = super().encode(token)

    def encode(self, token: str | None) -> torch.Tensor:
        """
        Returns one-hot encoded tensor for the token.

        Args:
            token: The token to encode.

        Returns:
            One-hot encoded tensor of shape (1, encoding_length).
        """
        if self.eval:
            if token is None or str(token) not in self.cache:
                self._count_for_unk_token += 1
                return torch.zeros(self.get_encoding_length(), dtype=torch.int64)
            index = self.cache[str(token)] + self.offset
            return torch.nn.functional.one_hot(
                torch.tensor(index), num_classes=self.get_encoding_length()
            )

        if token not in self.tokens_dict:
            self._count_for_unk_token += 1
            return torch.zeros(1, self.get_encoding_length(), dtype=torch.int64)

        return torch.nn.functional.one_hot(
            self.tokens_dict[token], num_classes=self.get_encoding_length()
        )


class AsIsEncoder(PropertyEncoder):
    """
    Returns the input value as it is, useful e.g. for float values.
    """

    @property
    def name(self) -> str:
        """Name of this encoder."""
        return "asis"

    def encode(self, token: float | int | None) -> torch.Tensor:
        """
        Return the input value as tensor, or zero tensor if None.

        Args:
            token: The value to encode.

        Returns:
            Tensor of shape (1,) containing the input value or zero.
        """
        if token is None:
            return torch.zeros(1, self.get_encoding_length())
        assert (
            len(token) == self.get_encoding_length()
        ), "Length of token should be equal to encoding length"
        # return torch.tensor([token]) # token is an ndarray, no need to create list of ndarray due to below warning
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        # (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_new.cpp:257.)
        # ----- fix: for above warning
        return torch.tensor(token).unsqueeze(0)  # shape: (1, len(token))


class BoolEncoder(PropertyEncoder):
    """
    Encodes boolean values as 0 or 1.
    """

    @property
    def name(self) -> str:
        """Name of this encoder."""
        return "bool"

    def encode(self, token: bool) -> torch.Tensor:
        """
        Encode boolean token as tensor.

        Args:
            token: Boolean value.

        Returns:
            Tensor with 1 if True else 0.
        """
        return torch.tensor([1 if token else 0])
