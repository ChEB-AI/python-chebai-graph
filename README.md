
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
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/csv_logger.yml --model=../python-chebai-graph/configs/model/gnn_res_gated.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce_weighted.yml
```

## Experiment Scripts

The scripts below reproduce the augmented-graph experiments introduced in this
work. They expect both repositories to be checked out as described in
[Recommended Folder Structure](#recommended-folder-structure), and they run
`python -m chebai` from the `python-chebai` checkout. Update the paths,
environment activation, and logging configuration in the scripts for a
different machine.

### Augmentation ablations

Each ablation script evaluates the same ten graph-construction variants:
`baseline`, `FGN`, `FGN+E`, `FGN+WGN`, `FGN+E+WGN`, `WGN`,
`gn_wall_fgwa_nfge`, `gn_wall_fgwa_wfge`, `gnwa_fgwa_nfge`, and
`gnwa_fgwa_wfge`. The experiments use seeds `0`, `42`, and `12345`, so each
dataset runs 30 configurations in total. The variants compare whether
functional-group nodes (FGN), functional-group edges (FGE), and the graph node
(WGN) are included, and whether their connections use the corresponding
weighted or unweighted schemes.

- [`aug-ablation-chebi.sh`](configs/data/augmented/aug-ablation/aug-ablation-chebi.sh)
  is a Slurm job-array runner for ChEBI50. It schedules the 10 configurations
  across the three seeds, trains the baseline GAT, and uses the ChEBI v252
  split.
- [`aug-ablation-tox21.sh`](configs/data/augmented/aug-ablation/aug-ablation-tox21.sh)
  runs the same 30-experiment matrix for Tox21 serially on one local GPU. It
  reports micro/macro F1 and ROC-AUC using the Tox21 split.
- [`aug-ablation-toxcast.sh`](configs/data/augmented/aug-ablation/aug-ablation-toxcast.sh)
  runs the corresponding ToxCast matrix serially on one local GPU. It uses
  ToxCast-specific input dimensions and learning rates and reports micro/macro
  F1 and ROC-AUC.

Run a local ablation script from this repository with, for example:

```bash
bash configs/data/augmented/aug-ablation/aug-ablation-tox21.sh
```

The ChEBI script is intended for Slurm and should be submitted with
`sbatch`. The Tox21 and ToxCast scripts run every seed and configuration in
sequence, rather than requiring Slurm.

### MoleculeNet datasets

[MoleculeNet](https://moleculenet.org/) is a collection of standard molecular
machine-learning benchmarks. These datasets cover binary classification tasks
with different label counts, class imbalance, and chemical domains, making
them useful for testing whether an architecture generalizes beyond ChEBI50.
The augmented-graph runner currently evaluates six MoleculeNet datasets:
**BBBP**, **BACE**, **SIDER**, **ClinTox**, **Tox21**, and **ToxCast**. Their
dataset configurations are in
`configs/data/augmented/final/<dataset>_final_augmented.yml`.

[`molnet.sh`](configs/data/augmented/final/molnet.sh) runs each of the six
datasets for seeds `0`, `42`, and `12345` on one local GPU. For each dataset
and seed it:

1. trains the augmented atom-motif-graph pooling GAT;
2. selects the best validation checkpoint using ROC-AUC;
3. evaluates that checkpoint on the test split; and
4. writes per-run metrics and checkpoint paths to
   `logs/molnet_test_results.tsv`, followed by mean and sample standard
   deviation summaries.

The script selects binary or multilabel trainers and metrics per dataset and
uses dataset-specific dropout and learning rates. Run it with:

```bash
bash configs/data/augmented/final/molnet.sh
```

For comparisons with published MoleculeNet results, use scaffold splits for
BBBP and BACE and random splits for Tox21, ToxCast, SIDER, and ClinTox. The
reported benchmark metric is the mean test ROC-AUC over three independent
runs; the runner also records the F1 metrics selected for each dataset. The
MoleculeNet experiments can be run with or without initializing the model from
the ZINC15 MSP checkpoint, allowing the contribution of pretraining to be
measured separately.

### Final ChEBI50 training

[`final-chebi.sh`](configs/data/augmented/final/final-chebi.sh) is a Slurm
job-array script that trains the final augmented-graph GAT on ChEBI50 v252 for
the three seeds. It uses the `gnwa_fgwa_wfge` configuration, the
atom-motif-graph pooling architecture, weighted binary cross-entropy, and the
standard micro/macro F1 metrics. Submit it with:

```bash
sbatch configs/data/augmented/final/final-chebi.sh
```

### ZINC15 pretraining

[`zinc_msp_pretraining.sh`](configs/data/augmented/pretraining/zinc_msp_pretraining.sh)
trains the augmented GAT with the masked-subgraph prediction (MSP)
pretraining objective on ZINC15. It uses the dedicated ZINC pretrainer,
`gat_msp.yml`, and the augmented ZINC15 data configuration. Validation batches
are disabled because this is a pretraining run; the resulting checkpoint can
then be used to initialize downstream augmented-graph experiments. The script
requests three GPUs on Slurm and is submitted with:

```bash
sbatch configs/data/augmented/pretraining/zinc_msp_pretraining.sh
```

## Augmented Graphs


Graph Neural Networks (GNNs) often fail to explicitly leverage the chemically meaningful substructures present within molecules (i.e. **functional groups (FGs)**).  To make this implicit information explicitly accessible to GNNs, we augment molecular graphs with **artificial nodes** that represent these substructures. The resulting graph are referred to as **augmented graphs**.
> Note: Rings are also treated as functional groups in our work.

In these augmented graphs, each functional group node is connected to the atoms
that constitute the group. Additionally, two functional group nodes are
connected if any atom belonging to one group shares a bond with an atom from
the other group. We also introduce a **graph node**, which is connected to all
atom nodes. Rings are treated as functional groups.

Functional-group nodes encode both the presence and the identity of the
detected group, rather than representing an unlabeled fragment. The graph node
acts as a global or master node: it provides a shortcut for information
exchange between distant parts of a molecule and collects a whole-molecule
representation.

### Best configuration: `GNWA_FGWA_WFGE`

Among the connection schemes evaluated, **GNWA_FGWA_WFGE** delivered the
strongest performance. The name describes the complete augmented topology:

- **GNWA** (*Graph Node With Atom connections*): the graph node is connected
  to every atom node.
- **FGWA** (*Functional Group Nodes With Atom connections*): every functional
  group node is connected to the atoms that belong to that group.
- **WFGE** (*With Functional Group Edges*): functional group nodes are
  connected when their member atoms are connected by a molecular bond.

This configuration is used by the final ChEBI50 and MoleculeNet experiments.
Its ChEBI50 data definition is
[`chebi50_final_augmented.yml`](configs/data/augmented/final/chebi50_final_augmented.yml),
and its atom-motif-graph pooling model is
[`amg_pool/gat.yml`](configs/model/augmented/pooling/amg_pool/gat.yml).
The topology is illustrated below.

The final augmented representation initializes atoms, bonds, functional-group
nodes, and the graph node with chemically meaningful, node-type-specific
properties. This is different from the isolated topology ablations, where
artificial nodes and added edges use zero initialization so that the effect of
connectivity can be measured independently of feature initialization and
type-specific pooling.

<img width="3488" height="1760" alt="mol graph drawio" src="https://github.com/user-attachments/assets/279dbd39-6656-45be-a66b-aa76ec0febdc" />

</br>
</br>

Below is the command for the model and data configuration that achieved the best classification performance using augmented graphs.

```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/wandb_logger.yml --model=../python-chebai-graph/configs/model/augmented/pooling/amg_pool/gat.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/augmented/properties/gnwa_fgwa_wfge/chebi50_all_props.yml --data.init_args.batch_size=64 --trainer.accumulate_grad_batches=1 --data.init_args.num_workers=10 --model.pass_loss_kwargs=false --data.init_args.chebi_version=252 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce_unweighted.yml --trainer.logger.init_args.name=gnwa_fgwa_wfge_final_42 --model.init_args.optimizer_kwargs.lr=0.002 --seed_everything=42 --data.init_args.splits_file_path=data/chebi_v252/ChEBI50/processed/splits.csv --trainer.logger.init_args.tags=["augmented_paper","final:gnwa_fgwa_wfge"]
```

### Model Hyperparameters

The model paths below are relative to the `python-chebai` working directory
used by the training commands. In this repository they correspond to the
following configuration files:

#### **GAT Architecture**

Choose one of the following GAT configurations:

- **Standard pooling**:
  [`configs/model/baselines/gat.yml`](configs/model/baselines/gat.yml)
  (`--model=../python-chebai-graph/configs/model/baselines/gat.yml`).
  `GATGraphPred` pools all node representations into one graph
  representation. This configuration is used for the ablation baselines.
- **Atom-augmented pooling**:
  [`configs/model/augmented/pooling/aa_pool/gat.yml`](configs/model/augmented/pooling/aa_pool/gat.yml).
  `GATAAPoolGraphPred` aggregates atom-node representations separately from
  artificial-node representations, then concatenates the two pooled vectors.
- **Atom-motif-graph pooling**:
  [`configs/model/augmented/pooling/amg_pool/gat.yml`](configs/model/augmented/pooling/amg_pool/gat.yml).
  `GATAMGPoolGraphPred` separately aggregates atom nodes, functional-group
  nodes, and the graph node before classification. This is the configuration
  used by the final augmented experiments.
- **Augmented graphs without separate pooling**:
  [`configs/model/augmented/pooling/no_pooling/gat.yml`](configs/model/augmented/pooling/no_pooling/gat.yml).
  This uses `GATGraphPred` with augmented node and edge features while pooling
  all nodes together.

The augmented GAT configurations use 203 node features and 12 edge features;
the baseline configuration uses 161 node features and 8 edge features. All
current GAT configurations default to four message-passing layers, 256 hidden
channels, 512 output channels, eight attention heads, and GATv2
(`v2: True`). The number of heads must divide the output or hidden channel
size. These values can be overridden, for example:

```bash
--model.config.num_layers=5
--model.config.heads=4
--model.config.v2=False
--model.config.dropout=0.1
```

Set `v2=False` to use the original `GATConv`; the default `v2=True` uses
`GATv2Conv`, which provides dynamic rather than static attention. For details,
see the [GATv2 paper](https://arxiv.org/abs/2105.14491).

The augmentation and property initialization do not add trainable parameters
to the core GNN. Type-specific pooling can increase the size of the
classification head, but the main performance gain is intended to come from
the augmented molecular representation and its chemically meaningful
features.

#### **ResGated Architecture**

Choose one of the following ResGated configurations:

- **Atom-motif-graph pooling**:
  [`configs/model/res_aug_amgpool.yml`](configs/model/res_aug_amgpool.yml).
- **Atom-augmented pooling**:
  [`configs/model/res_aug_aapool.yml`](configs/model/res_aug_aapool.yml).

Both augmented ResGated configurations use 203 node features, 12 edge
features, four message-passing layers, 256 hidden channels, and 512 output
channels. The ResGated convolution does not expose GAT attention-head or
`v2` settings.

#### **Shared hyperparameters**

These settings are available for both GAT and ResGated model configurations:

- **Dropout**: `--model.config.dropout=0.1` (default: `0`)
- **Number of final linear layers**: `--model.n_linear_layers=2`
  (default: `1`)

The complete defaults for each architecture are defined directly in the
linked YAML files above.

## Random Node Initialization

### Static Node Initialization

In this type of node initialization, the node features (and/or edge features) of the given molecular graph are initialized only once during dataset creation with the given initialization scheme.

```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/wandb_logger.yml --model=../python-chebai-graph/configs/model/resgated.yml --model.config.in_channels=203 --model.config.edge_dim=11 --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --data.pad_node_features=45 --data.pad_edge_features=4 --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --data.init_args.persistent_workers=False --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce_weighted.yml --trainer.logger.init_args.name=gni_res_props+zeros_s0
```

In the above command, for each node we use the 158 node features (corresponding the node properties defined in `chebi50_graph_properties.yml`) which are retrieved from RDKit and additional 45 additional features (specified by `--data.pad_node_features=45`) drawn from a normal distribution (default).

You can change the distribution from which additional are drawn by using the following config in above command: `--data.distribution=zeros`

Available distributions: `"normal", "uniform", "xavier_normal", "xavier_uniform", "zeros"`


Similarly, each edge is initialized with 7 RDKit features and 4 additional features drawn from the given distribution.


If you want all node (and edge) features to be drawn from a given distribution (i.e., ignore RDKit features), use: `--data=../python-chebai-graph/configs/data/chebi50_static_gni.yml`


Refer to the data class code for details.


### Dynamic Node Initialization

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
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/wandb_logger.yml --model=../python-chebai-graph/configs/model/resgated_dynamic_gni.yml --model.config.in_channels=203 --model.config.edge_dim=11 --model.config.complete_randomness=False --model.config.pad_node_features=45 --model.config.pad_edge_features=4 --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --data.init_args.batch_size=128 --trainer.accumulate_grad_batches=4 --data.init_args.num_workers=10 --data.init_args.persistent_workers=False --model.pass_loss_kwargs=false --data.init_args.chebi_version=241 --trainer.min_epochs=200 --trainer.max_epochs=200 --model.criterion=configs/loss/bce_weighted.yml --trainer.logger.init_args.name=gni_dres_props+rand_s0
```
