

## Installation

Some requirements may not be installed successfully automatically. 
To install the `torch-` libraries, use

`pip install torch-${lib} -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html`

where `${lib}` is either `scatter`, `geometric`, `sparse` or `cluster`, and
`${CUDA}` is either `cpu`, `cu118` or `cu121` (depending on your system, see e.g. 
[torch-geometric docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))