## Unsupervised Learning Approach for 3D Shape Alignment (PyTorch)

This repository provides a PyTorch implementation for learning correspondence between 3D shapes having undergone non-rigid deformations.

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Installing PyTorch may require an ad hoc procedure, depending on your computer settings.

### Data & Pre-Trained weights and biases
Due to the enormous size of the data, anyone can contact with for the same via the mentioned contact email

The trained model weights and the biases for the basis and decriptor model have been provided in the repository 
```
python .\model\pretraned_weights\basis_model_best_mod_select_epoch_63.pth
python .\models\pretrained_weights\desc_model_best_C_0.01_epoch_62.pth
```
### Training

To train the basis and descriptors models with any specific dataset, change the paths provided for the datasets and run these commands:

```train
python .\code\train_basis.py
python .\code\train_desc.py
```

### Evaluation

To evaluate the model on FAUST, run:

```eval
python .\evaluation\evaluation_faust.py
```
And in matlab the script:
```eval
.\evaluation\eval_graph.m
.\evaluation\plot_perf.m
```
### Results

These are the results of the implementations:

| Model name         | Ours            |   Supervised       |
| ------------------ |---------------- | -------------- |
| PyTorch            |     6.0e-2      |      6.78e-2    |

The idea and the code has driven it's inspiration from the below mentioned paper

@article{marin2020correspondence,
  title={Correspondence learning via linearly-invariant embedding},
  author={Marin, Riccardo and Rakotosaona, Marie-Julie and Melzi, Simone and Ovsjanikov, Maks},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
