# Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks

The code is tested under Python 3.6 and PyTorch 1.4.0. It automatically downloads the pretrained models (either VGG-16-BN or ResNet-50) and requires access to ImageNet validation set.

The following are examples of how to run the attacks in the different threat models.

## Sparse-RS attacks
### L0-bounded
In this case `k` represents the number of pixels to modify. For untargeted attacks
```
CUDA_VISIBLE_DEVICES=0 python eval.py --norm=L0 \
	--model=[pt_vgg | pt_resnet] --n_queries=10000 --alpha_init=0.3 \
	--data_path=/path/to/validation/set --k=150 --n_ex=500
```
and for targeted attacks please use `--targeted --n_queries=100000 --alpha_init=0.1`. The target class is randomly chosen for each point.

As additional options the flag `--constant_schedule` uses a constant schedule for `alpha` instead of the piecewise constant decreasing one, while with `--seed=N` it is possible to set a custom random seed.

### Image-specific patches
For image- and location-specific patches of size 30x30 (with `k=900`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py --norm=patches \
	--model=[pt_vgg | pt_resnet] --n_queries=10000 --alpha_init=0.3 \
	--data_path=/path/to/validation/set --k=900 --n_ex=100
```

### Universal patches and frames
For universal untargeted patches of size 50x50 (with `k=2500`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--norm=patches_universal --model=[pt_vgg | pt_resnet] \
	--n_queries=100000 --alpha_init=0.3 \
	--data_path=/path/to/validation/set --k=2500 --n_ex=100
```
while for universal untargeted frames of width 4 (with `k=4`)
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--norm=frames_universal --model=[pt_vgg | pt_resnet] \
	--n_queries=100000 --alpha_init=0.005 \
	--data_path=/path/to/validation/set --k=4 --n_ex=100
```
For **universal targeted** attacks add at the previous commands `--targeted --target_class=920` with the number corresponding to the target label.

## Visualizing resulting images
We provide a script `vis_images.py` to visualize the images produced by the attacks. To use it please run

```python vis_images --path_data=/path/to/saved/results```