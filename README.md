

## Installation

Some requirements may not be installed successfully automatically. 
To install the `torch-` libraries, use

`pip install torch-${lib} -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html`

where `${lib}` is either `scatter`, `geometric`, `sparse` or `cluster`, and
`${CUDA}` is either `cpu`, `cu118` or `cu121` (depending on your system, see e.g. 
[torch-geometric docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html))


## Commands

Pretraining on a atom / bond masking task with PubChem data (feature-branch):
```
python3 -m chebai fit --model=../chebai-graph/configs/model/gnn_resgated_pretrain.yml --data=../chebai-graph/configs/data/pubchem_graph.yml --trainer=configs/training/pretraining_trainer.yml
```

Training on the ontology prediction task (here for ChEBI50, v227, 200 epochs)
```
python3 -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.callbacks=configs/training/default_callbacks.yml --model=../chebai-graph/configs/model/gnn_res_gated.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --data=../chebai-graph/configs/data/chebi50_graph_properties.yml --model.out_dim=1446 --model.criterion=configs/loss/bce.yml --data.init_args.batch_size=40 --trainer.logger.init_args.name=chebi50_bce_unweighted_resgatedgraph --data.init_args.num_workers=12 --model.pass_loss_kwargs=false --data.init_args.chebi_version=227 --trainer.min_epochs=200 --trainer.max_epochs=200
```
