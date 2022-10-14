# Augmented Equivariant Attention Networks for Microscopy Image Transformation

This is the official implementation of AEANet in the paper [Augmented Equivariant Attention Networks for Microscopy Image Transformation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9785968) accepted by TMI.

<p align="center">
  <img src="ba-lib.png" alt="drawing" width=70%/>
</p>

## System Requirements and Environmental setup

We use the same system and environment as the work GVTNets. You can follow the insturctions in [this repo](https://github.com/divelab/GVTNets/) to setup the enviroment.

## Usage

You can follow the example in this [Jypyter notebook](https://github.com/divelab/AEANets/blob/main/model3_pooled_batchatt_lib_kshape.ipynb) for the training and prediction using AEANets to reproduce the results in our paper or on your own data. We have provided the training checkpoints to reproduce the results.

To train the model, simply uncomment the following line in Cell 3

```
model.train(sources, targets, [256,256], validation=None, steps=120000, batch_size=8,seed=1)
```

The arguments for the `model.train()` methods include

- source_lst: a numpy array of training low-quality images of shape [N, W, H, C].
- target_lst: a numpy array of training high-quality images of shape [N, W, H, C], with in same order to source images.
- patch_size: the patch size used for training. Will randomly crop training images into patches.
- validation: [Optional] A tuple of (source, target) pair as the validation set.

To evaluate the model, use the `evaluate_mean_wh` function. Specifically, `evaluate_mean_wh(None)` performs prediction on the entire given image and generally produces the best results. Please refer to Figure 9 in Appendix VII for more discussions.


## Bibtex

If you use our code, please consider cite our paper

```
@article{xie2022augmented,
  title={Augmented Equivariant Attention Networks for Microscopy Image Transformation},
  author={Xie, Yaochen and Ding, Yu and Ji, Shuiwang},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
```
