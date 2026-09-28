"""
Overbagging (REMEDIAL resampling, bagging and ML-ROS oversampling) for graph datasets.

The three steps are chained via intermediate files in `processed_dir_main`:
    1. `*_Resampled`:    data.pkl -> data_resampled.pkl (REMEDIAL)
    2. `*_Bootstrapped`: `input_data_file` (e.g. data_resampled.pkl) -> data_{bag_name}.pkl
    3. `*_MLROS`:        `add_to_file` + samples from `take_from_file` -> {add_to_file}_oversampled_with_{rate}_from_{take_from_file}.pkl

All intermediate datasets only contain molecules from data.pkl, so the molecular properties are always computed
from data.pkl and shared between all datasets (and the regular dataset) in `processed_dir/properties`.
"""

from chebai.preprocessing.datasets.ml_overbagging import (
    _BootstrapDynamicDataset,
    _MLROSDynamicDataset,
    _ResampledDynamicDataset,
)

from .chebi import ChEBI25_WFGE_WGN_AsPerNodeType


class _OverbaggingPropertiesMixIn:
    @property
    def property_source_file_names(self) -> list[str]:
        # data.pkl contains every molecule exactly once (resampled / bagged files contain duplicate idents)
        return [self._data_pkl_filename]


class ChEBI25_WFGE_WGN_AsPerNodeType_Resampled(
    _OverbaggingPropertiesMixIn,
    _ResampledDynamicDataset,
    ChEBI25_WFGE_WGN_AsPerNodeType,
):
    pass


class ChEBI25_WFGE_WGN_AsPerNodeType_Bootstrapped(
    _OverbaggingPropertiesMixIn,
    _BootstrapDynamicDataset,
    ChEBI25_WFGE_WGN_AsPerNodeType,
):
    pass


class ChEBI25_WFGE_WGN_AsPerNodeType_MLROS(
    _OverbaggingPropertiesMixIn,
    _MLROSDynamicDataset,
    ChEBI25_WFGE_WGN_AsPerNodeType,
):
    pass
