import abc
import os


class PropertyEncoder(abc.ABC):
    def __init__(self, property, **kwargs):
        self.property = property

    def get_encoding_length(self) -> int:
        return 1

    def encode(self, value):
        raise NotImplementedError

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
        return self.cache.index(str(token)) + self.offset
