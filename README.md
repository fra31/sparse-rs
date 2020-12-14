# Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks
**Francesco Croce, Maksym Andriushchenko, Naman D. Singh, Nicolas Flammarion, Matthias Hein**

**University of Tübingen and EPFL**

**Paper:** [https://arxiv.org/abs/2006.12834](https://arxiv.org/abs/2006.12834)

**A short version is accepted to [ECCV'20 Workshop on Adversarial Robustness in the Real World](https://eccv20-adv-workshop.github.io/)**

## Abstract
A large body of research has focused on adversarial attacks which require to modify all input features with small L2- or Linf-norms.
In this paper we instead focus on query-efficient *sparse* attacks in the black-box setting. Our versatile framework, **Sparse-RS**, based on random search
achieves state-of-the-art success rate and query efficiency for different sparse attack models such as L0-bounded perturbations (outperforming established white-box methods), adversarial patches, and adversarial framing. 
We show the effectiveness of **Sparse-RS** on different datasets considering problems from image recognition and malware detection and multiple variations of sparse threat models, including targeted and universal perturbations. 
In particular **Sparse-RS** can be used for realistic attacks such as universal adversarial patch attacks without requiring a substitute model. 



## About the paper
Our proposed **Sparse-RS** framework is based on random search. Its main advantages are its simplicity and its wide applicability 
to multiple threat models:
<p align="center"><img src="img/algorithm_sparse_rs.png" width="700"></p>

We illustrate the versatility of the **Sparse-RS** framework by generating universal targeted patches/frames, and also
image- and location-specific patches:
<p align="center"><img src="img/universal_targeted_patches_frames.png" width="700"></p>
<p align="center"><img src="img/image_and_location_specific_patches.png" width="700"></p>

In all these threat models we improve over other approaches:
<p align="center"><img src="img/universal_targeted_patches_frames_table.png" width="700"></p>
<p align="center"><img src="img/image_and_location_specific_patches_table.png" width="700"></p>

Moreover, for L0-perturbations **Sparse-RS** can even outperform strong white-box baselines such as L0 PGD (see PGD_0 (wb)).
<p align="center"><img src="img/l0_curves.png" width="700"></p>



## Code of Sparse-RS
The code is tested under Python 3.8.5 and PyTorch 1.8.0. It automatically downloads the pretrained models (either VGG-16-BN or ResNet-50) and requires access to ImageNet validation set.

The following are examples of how to run the attacks in the different threat models.

### L0-bounded
In this case `k` represents the number of pixels to modify. For untargeted attacks
```
CUDA_VISIBLE_DEVICES=0 python eval.py --norm=L0 \
	--model=[pt_vgg | pt_resnet] --n_queries=10000 --alpha_init=0.3 \
	--data_path=/path/to/validation/set --k=150 --n_ex=500
```
and for targeted attacks please use `--targeted --n_queries=100000 --alpha_init=0.1`. The target class is randomly chosen for each point.

As additional options the flag `--constant_schedule` uses a constant schedule for `alpha` instead of the piecewise constant decreasing one, while with `--seed=N` it is possible to set a custom random seed.

### Image-specific patches and frames
For untargeted image- and location-specific patches of size 20x20 (with `k=400`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py --norm=patches \
	--model=[pt_vgg | pt_resnet] --n_queries=10000 --alpha_init=0.4 \
	--data_path=/path/to/validation/set --k=400 --n_ex=100
```

For targeted patches (size 40x40) please use `--targeted --n_queries=50000 --alpha_init=0.1 --k=1600`. The target class is randomly chosen for each point.

For untargeted image-specific frames of width 2 pixels (with `k=2`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py --norm=frames \
	--model=[pt_vgg | pt_resnet] --n_queries=10000 --alpha_init=0.5 \
	--data_path=/path/to/validation/set --k=2 --n_ex=100
```

For targeted frames (width of 3 pixels) please use `--targeted --n_queries=50000 --alpha_init=0.5 --k=3`. The target class is randomly chosen for each point.

### Universal patches and frames
For targeted universal patches of size 50x50 (with `k=2500`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--norm=patches_universal --model=[pt_vgg | pt_resnet] \
	--n_queries=100000 --alpha_init=0.3 \
	--data_path=/path/to/validation/set --k=2500 \
	--n_ex=30 --targeted --target_class=530
```

and for targeted universal frames of width 6 pixels (`k=6`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--norm=frames_universal --model=[pt_vgg | pt_resnet] \
	--n_queries=100000 --alpha_init=1.667 \
	--data_path=/path/to/validation/set --k=6 \
	--n_ex=30 --targeted --target_class=530
```
The argument `--target_class` specifies the number corresponding to the target label. To generate universal attacks we use batches of 30 images resampled every 10000 queries.

## Visualizing resulting images
We provide a script `vis_images.py` to visualize the images produced by the attacks. To use it please run

```python vis_images --path_data=/path/to/saved/results```
