
# ChEB-AI Graph

Graph-based models for molecular property prediction and ontology classification, built on top of the [`python-chebai`](https://github.com/ChEB-AI/python-chebai) codebase.



## Installation

To install this repository, download it and run 

```bash
pip install .
```

or install it directly with
```bash
pip install git+https://github.com/ChEB-AI/python-chebai-graph.git
```

The dependencies `torch`, `torch_geometric` and `torch-sparse` cannot be installed automatically.

Use the following command:

```bash
pip install torch torch_scatter torch_geometric -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Replace:
- `${TORCH}` with a PyTorch version (e.g., `2.6.0`; for later versions, check first if they are compatible with torch_scatter and torch_geometric)
- `${CUDA}` with e.g. `cpu`, `cu118`, or `cu121` depending on your system and CUDA version

If you already have `torch` installed, make sure that `torch_scatter` and `torch_geometric` are compatible with your 
PyTorch version and are installed with the same CUDA version.

For a full list of currently available PyTorch versions and CUDA compatibility, please refer to libraries' official documentation:
- [torch](https://pytorch.org/get-started/locally/)
- [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation)
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)

_Note for developers_: If you want to install the package in editable mode, use the following command instead:

```bash
pip install -e .
```

## Recommended Folder Structure

ChEB-AI Graph is not a standalone library. Instead, it provides additional models and datasets for [`python-chebai`](https://github.com/ChEB-AI/python-chebai).
The training relies on config files that are located either in `python-chebai` or in this repository.

Therefore, for training, we recommend to clone both repositories into a common parent directory. For instance, your project can look like this:

```
my_projects/
├── python-chebai/
│   ├── chebai/
│   ├── configs/
│   └── ...
└── python-chebai-graph/
    ├── chebai_graph/
    ├── configs/
    └── ...
```

## Training & Pretraining

### Ontology Prediction


This example command trains a Residual Gated Graph Convolutional Network on the ChEBI50 dataset (see [wiki](https://github.com/ChEB-AI/python-chebai/wiki/Data-Management)). 
The dataset has a customizable list of properties for atoms, bonds and molecules that are added to the graph. 
The list can be found in the `configs/data/chebi50_graph_properties.yml` file.

```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/csv_logger.yml --model=../python-chebai-graph/configs/model/gnn_res_gated.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce.yml
```

