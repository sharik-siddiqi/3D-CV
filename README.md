## Unsupervised Learning Approach for 3D Shape Alignment (PyTorch)

This repository provides a PyTorch implementation for learning correspondence between 3D shapes having undergone non-rigid deformations.

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Installing PyTorch may require an ad hoc procedure, depending on your computer settings.

### Data 
Due to the enormous size of the data, anyone can contact me for obtaining the same via the mentioned contact email-id

The dataset used here involves 1000 low density meshes (2100/1000 vertices) taken from the **SURREAL** dataset. The dataset has been generated using the **SMPL** model with various shape and pose parameters. 

In various locations, you can modify the code accordding to the dataset generated. The modifications are as follows:

1) **train_basis.py** : 
- Replace all the supposed mat file locations given in the form *'./pyshot/shot_faust_0_2100_sam.mat'* by files containing the SHOT descriptors for your dataset.
- Replace all the supposed mat file locations given in the form *'./pyshot/hks_surreal_0_2100_sam.mat'* by files containing the HKS descriptors for your dataset.
- Replace all the supposed mat file locations given in the form *'./pyshot/wks_surreal_0_2100_sam.mat'* by files containing the WKS descriptors for your dataset.

2) **dataloader_light_rand.py** :
- Replace all the supposed mat file locations given in the form *'./data/unsup_data_t_0.005_hk_0_same.mat'* by files containing the indices and heat kernels (at desired t values) for your dataset.
- Replace *"/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/match_6890_2100 (1).mat"* (6890 -> 2100) & *"/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/match_2100_1000 (1).mat"* (2100->1000) by files containing correspondences between high density and low density meshes in order to work with the latter
- Replace *"/home/raml_sharik/Diff-FMAPs-PyTorch-main/data/3973_simp_1000.obj"* by the location of the simplified mesh

### Model Architecture
The *'model.py'* file contains the model architecture for both basis and descriptor models

### Pre-Trained weights and biases
The trained model weights and the biases for the basis and decriptor model have been provided in the repository 
```
python .\model\pretraned_weights\basis_model_best_mod_select_epoch_63.pth
python .\models\pretrained_weights\desc_model_best_C_0.01_epoch_62.pth
```
### Training

To train the basis and descriptors models with any specific dataset (after making necessary changes), run these commands:

```train
python .\code\train_basis.py
python .\code\train_desc.py
```

### Evaluation

To evaluate the model on FAUST, run:
```eval
python .\evaluation\evaluation_faust.py
```

After getting the files *'curve_geo_error_non_iso.mat'* and *'dd2_non_iso.mat'* from the above matlab script, run the matlab scripts:
```eval
.\evaluation\eval_graph.m
.\evaluation\plot_perf.m
```

### Results

These are the results of the implementations in terms of mean error per vertex of our test dataset:

| Model name         | Ours            |   Supervised       |
| ------------------ |---------------- | ------------------ |
| PyTorch            |     6.0e-2      |      6.78e-2       |

The idea and the code has driven it's inspiration from the below mentioned paper

@article{marin2020correspondence,
  title={Correspondence learning via linearly-invariant embedding},
  author={Marin, Riccardo and Rakotosaona, Marie-Julie and Melzi, Simone and Ovsjanikov, Maks},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
