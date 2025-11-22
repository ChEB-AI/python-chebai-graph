
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

The dependencies `torch`, `torch_geometric` and `torch_scatter` cannot be installed automatically.

Use the following command:

```bash
pip install torch torch_scatter torch_geometric -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Replace:
- `${TORCH}` with a PyTorch version (e.g., `2.8.0`; for later versions, check first if they are compatible with torch_scatter and [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))
- `${CUDA}` with `cpu`, `cu118`, `cu121` (or other, depending on your system and CUDA version)

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

## Augmented Graphs

Below is the command for the model and data configuration that achieved the best classification performance using augmented graphs.


```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/wandb_logger.yml --model=../python-chebai-graph/configs/model/gat_aug_amgpool.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --model.config.v2=True --data=../python-chebai-graph/configs/data/chebi50_aug_prop_as_per_node.yml --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce.yml --trainer.logger.init_args.name=gatv2_amg_s0
```

### Model Hyperparameters

#### **GAT Architecture**

To use a GAT-based model, choose **one** of the following configs:

- **Atom–Motif–Graph Node Pooling**: `--model=../python-chebai-graph/configs/model/gat_aug_amgpool.yml`
- **Atom-Augmented Node Pooling**: `--model=../python-chebai-graph/configs/model/gat_aug_aagpool.yml`
- **Standard Pooling**: `--model=../python-chebai-graph/configs/model/gat.yml`

#### GAT-specific hyperparameters

- **Number of message-passing layers**: `--model.config.num_layers=5`        (default: 4)
- **Attention heads**: `--model.config.heads=4`             (default: 8)
  > Note: The number of heads should be divisible by the output channels (or hidden channels if output channels are not specified).
- **Use GATv2**: `--model.config.v2=True`             (default: False)

#### **ResGated Architecture**

To use a ResGated GNN model, choose **one** of the following configs:

- **Atom–Motif–Graph Node Pooling**: `--model=../python-chebai-graph/configs/model/res_aug_amgpool.yml`
- **Atom-Augmented Node Pooling**: `--model=../python-chebai-graph/configs/model/res_aug_aagpool.yml`
- **Standard Pooling**: `--model=../python-chebai-graph/configs/model/resgated.yml`

#### **Common Hyperparameters**

These can be used for both GAT and ResGated architectures:

- **Dropout**: `--model.config.dropout=0.1`         (default: 0)
- **Number of final linear layers**: `--model.n_linear_layers=2`         (default: 1)

# Random Node Initialization

## Static Node Initialization

In this type of node initialization, the node features (and/or edge features) of the given molecular graph are initialized only once during dataset creation with the given initialization scheme.

```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/wandb_logger.yml --model=../python-chebai-graph/configs/model/resgated.yml --model.config.in_channels=203 --model.config.edge_dim=11 --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --data.pad_node_features=45 --data.pad_edge_features=4 --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --data.init_args.persistent_workers=False --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce.yml --trainer.logger.init_args.name=gni_res_props+zeros_s0
```

In the above command, for each node we use the 158 node features (corresponding the node properties defined in `chebi50_graph_properties.yml`) which are retrieved from RDKit and add 45 additional features (specified by `--data.pad_node_features=45`) drawn from a normal distribution (default).

You can change the distribution using the following config in above command: `--data.distribution=zeros`

Available distributions: `"normal", "uniform", "xavier_normal", "xavier_uniform", "zeros"`


Similarly, each edge is initialized with 7 RDKit features and 4 additional features drawn from the given distribution.


If you want all node (and edge) features to be drawn from a given distribution (i.e., ignore RDKit features), use: `--data=../python-chebai-graph/configs/data/chebi50_static_gni.yml`


Refer to the data class code for details.


## Dynamic Node Initialization

In this type of node initialization, the node features (and/or edge features) of the molecular graph are initialized at **each forward pass** of the model using the given initialization scheme.



Currently, dynamic node initialization is implemented only for the **resgated** architecture by specifying: `--model=../python-chebai-graph/configs/model/resgated_dynamic_gni.yml`

To keep RDKit features and *add* dynamically initialized features use the following config in the command:

```
--model.config.complete_randomness=False
--model.config.pad_node_features=45
```

The additional features are drawn from normal distribution (default). You can change it using:`--model.config.distribution=uniform`

If all features should be initialized from the given distribution, remove the complete_randomness flag (default is True).


Please find below the command for a typical dynamic node initialization:

```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/wandb_logger.yml --model=../python-chebai-graph/configs/model/resgated_dynamic_gni.yml --model.config.in_channels=203 --model.config.edge_dim=11 --model.config.complete_randomness=False --model.config.pad_node_features=45 --model.config.pad_edge_features=4 --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --data.init_args.persistent_workers=False --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce.yml --trainer.logger.init_args.name=gni_dres_props+rand_s0
```
