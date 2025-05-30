
# ChEB-AI Graph

Graph-based models for molecular property prediction and ontology classification, built on top of the [`python-chebai`](https://github.com/ChEB-AI/python-chebai) codebase.



## Installation

Some dependencies, especially `torch-` libraries, may not install automatically. In case you are experiencing problems, please install them manually **with versions compatible with your installed PyTorch version**.

Use the following command:

```bash
pip install torch_scatter torch_geometric torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Replace:
- `${TORCH}` with your installed PyTorch version (e.g., `2.6.0`)
- `${CUDA}` with: `cpu`, `cu118`, or `cu121` depending on your system and CUDA version

See also [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## Recommended Folder Structure

ChEB-AI Graph is not a standalone library. Instead, it provides additional models and datasets for [`python-chebai`](https://github.com/ChEB-AI/python-chebai).

Therefore, for training we recommend to clone both repositories into a common parent directory. For instance, your project can look like this:

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

### Important Note

- Before executing the following commands, ensure you are in the `python-chebai` directory and have set the `PYTHONPATH` to the `python-chebai-graph` directory, as explained in the [PYTHONPATH Explained](#-pythonpath-explained) section below.
- To avoid any potential error, we recommend **configuring both directories** in the `PYTHONPATH`, using following command (use **semicolon (`;`)** on Windows, and **colon (`:`)** on Linux as a separator)
  ```bash
      set PYTHONPATH=path/to/python-chebai;path/to/python-chebai-graph
  ```

  

### 🧠 Pretraining (Atom/Bond Masking on PubChem)

```bash
python -m chebai fit --model=../python-chebai-graph/configs/model/gnn_resgated_pretrain.yml --data=../python-chebai-graph/configs/data/pubchem_graph.yml --trainer=configs/training/pretraining_trainer.yml
```


### 📊 Ontology Prediction (ChEBI50, v231, 200 epochs)

This command trains a Residual Gated Graph Convolutional Network on the ChEBI50 dataset (see [wiki](https://github.com/ChEB-AI/python-chebai/wiki/Data-Management)). 
The dataset has a customizable list of properties for atoms, bonds and molecules that are added to the graph. 
The list can be found in the `configs/data/chebi50_graph_properties.yml` file. 

```bash
python -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.callbacks=configs/training/default_callbacks.yml --model=../python-chebai-graph/configs/model/gnn_res_gated.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --data=../python-chebai-graph/configs/data/chebi50_graph_properties.yml --model.criterion=configs/loss/bce.yml --data.init_args.batch_size=40 --data.init_args.num_workers=12 --data.init_args.chebi_version=231 --trainer.logger.init_args.name=chebi50_bce_unweighted_resgatedgraph --trainer.min_epochs=200 --trainer.max_epochs=200 --model.pass_loss_kwargs=false
```



## 🧭 PYTHONPATH Explained

### What is `PYTHONPATH`?

`PYTHONPATH` is an environment variable that tells Python where to search for modules that aren't installed via `pip` or not in your current working directory.

### Why You Need It

If your config refers to a custom module like:

```yaml
class_path: chebai_graph.preprocessing.datasets.chebi.ChEBI50GraphData
```

...and you're running the code from `python-chebai`, Python won't know where to find `chebai_graph` (from another repo like `python-chebai-graph/`) unless you add it to `PYTHONPATH`.


### How Python Finds Modules

Python looks for imports in this order:

1. Current directory
2. Standard library
3. Paths in `PYTHONPATH`
4. Installed packages (`site-packages`)

You can inspect the full search paths:

```bash
python -c "import sys; print(sys.path)"
```



### ✅ Setting `PYTHONPATH`

#### 🐧 Linux / macOS

```bash
export PYTHONPATH=/path/to/python-chebai-graph
echo $PYTHONPATH
```

#### 🪟 Windows CMD

```cmd
set PYTHONPATH=C:\path\to\python-chebai-graph
echo %PYTHONPATH%
```

> 💡 Note: This is temporary for your terminal session. To make it permanent, add it to your system environment variables.
