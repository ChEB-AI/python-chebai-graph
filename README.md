
# 🧪 ChEB-AI Graph

Graph-based models for molecular property prediction and ontology classification, built on top of the [`python-chebai`](https://github.com/ChEB-AI/python-chebai) codebase.



## 🔧 Installation

Some dependencies, especially `torch-` libraries, may not install automatically. You should install them manually **with versions compatible with your installed PyTorch version**, or you may encounter unexpected errors.

Use the following command:

```bash
pip install torch-${lib} -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Replace:

- `${lib}` with one of: `scatter`, `geometric`, `sparse`, or `cluster`
- `${TORCH}` with your installed PyTorch version (e.g., `2.6.0`)
- `${CUDA}` with: `cpu`, `cu118`, or `cu121` depending on your system and CUDA version

### ❗ Version Compatibility Note

**Ensure that the torch-scatter, torch-geometric, etc., versions are compatible with your installed PyTorch version.**  
>Inconsistencies between your installed PyTorch version and the versions of torch-scatter, torch-geometric, and related libraries can cause unexpected or strange errors at runtime.
For example, if you have:

```bash
torch==2.6.0
```

Then use:

```bash
pip install torch-${lib} -f https://data.pyg.org/whl/torch-2.6.0+${CUDA}.html
```

Always ensure the installed library versions match your exact `torch` version.

📚 Refer to the [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for full compatibility instructions.



## 🗂 Recommended Folder Structure

To combine configuration files from both `python-chebai` and `python-chebai-graph`, structure your project like this:

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

This setup enables shared access to data and model configurations for both Transformer and GNN-based models.



## 🚀 Training & Pretraining

### ⚠️ Important Note

- Before executing the following commands, ensure you are in the `python-chebai` directory and have set the `PYTHONPATH` to the `python-chebai-graph` directory, as explained in the [PYTHONPATH Explained](#-pythonpath-explained) section below.
- To avoid any potential error, we recommend **configuring both directories** in the `PYTHONPATH`, using following command (for windows)
  ```bash
      set PYTHONPATH=path/to/python-chebai;path/to/python-chebai-graph
  ```

  

### 🧠 Pretraining (Atom/Bond Masking on PubChem)

```bash
python -m chebai fit --model=../python-chebai-graph/configs/model/gnn_resgated_pretrain.yml --data=../python-chebai-graph/configs/data/pubchem_graph.yml --trainer=configs/training/pretraining_trainer.yml
```


### 📊 Ontology Prediction (ChEBI50, v231, 200 epochs)

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
export PYTHONPATH=/path/to/python-chebai
echo $PYTHONPATH
```

#### 🪟 Windows CMD

```cmd
set PYTHONPATH=C:\path\to\python-chebai
echo %PYTHONPATH%
```

> 💡 Note: This is temporary for your terminal session. To make it permanent, add it to your system environment variables.
