

## Installation

Some requirements may not be installed successfully automatically.
To install the `torch-` libraries, use

`pip install torch-${lib} -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html`

where `${lib}` is either `scatter`, `geometric`, `sparse` or `cluster`, and
`${CUDA}` is either `cpu`, `cu118` or `cu121` (depending on your system, see e.g.
[torch-geometric docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))


## Commands

For training, config files from the `python-chebai` and `python-chebai-graph` repositories can be combined. This requires that you download the [source code of python-chebai](https://github.com/ChEB-AI/python-chebai). Make sure that you are in the right folder and know the relative path to the other repository.

We recommend the following setup:

  my_projects
    python-chebai
      chebai
      configs
      data
      ...
    python-chebai-graph
      chebai_graph
      configs
      ...

  If you run the command from the `python-chebai` directory, you can use the same data for both chebai- and chebai-graph-models (e.g., Transformers and GNNs).
  Then you have to use `{path-to-chebai} -> .` and `{path-to-chebai-graph} -> ../python-chebai-graph`.

Pretraining on a atom / bond masking task with PubChem data (feature-branch):
```
python3 -m chebai fit --model={path-to-chebai-graph}/configs/model/gnn_resgated_pretrain.yml --data={path-to-chebai-graph}/configs/data/pubchem_graph.yml --trainer={path-to-chebai}/configs/training/pretraining_trainer.yml
```

Training on the ontology prediction task (here for ChEBI50, v231, 200 epochs)
```
python3 -m chebai fit --trainer={path-to-chebai}/configs/training/default_trainer.yml --trainer.callbacks={path-to-chebai}/configs/training/default_callbacks.yml --model={path-to-chebai-graph}/configs/model/gnn_res_gated.yml --model.train_metrics={path-to-chebai}/configs/metrics/micro-macro-f1.yml --model.test_metrics={path-to-chebai}/configs/metrics/micro-macro-f1.yml --model.val_metrics={path-to-chebai}/configs/metrics/micro-macro-f1.yml --data={path-to-chebai-graph}/configs/data/chebi50_graph_properties.yml --model.criterion=c{path-to-chebai}/onfigs/loss/bce.yml --data.init_args.batch_size=40 --trainer.logger.init_args.name=chebi50_bce_unweighted_resgatedgraph --data.init_args.num_workers=12 --model.pass_loss_kwargs=false --data.init_args.chebi_version=231 --trainer.min_epochs=200 --trainer.max_epochs=200
```
