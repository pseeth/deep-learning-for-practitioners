# Batch Normalization

## Resources
* Original Paper: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
* Andrew Ng Videos
  * [Normalizing Activations in a Networks](https://www.youtube.com/watch?v=tNIpEZLv_eg)
  * [Why does batch normalization work?](https://www.youtube.com/watch?v=nUUqwaxLnWs)
  * [Batch norm at test time](https://www.youtube.com/watch?v=5qefnAek8OA)
* Presenting Paper: [How Batch Normalization Works](https://arxiv.org/pdf/1805.11604.pdf)
  * [Overview Video](https://youtu.be/ZOabsYbmBRM)
* [PyTorch source code for Batch Norm](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d)


## Running Example Code:
The python package `bn` contains two experiments to explore batch normalization. One with a simple linear model that trains over MNIST and second that trains using a VGG11 model on CIFAR. The main purpose of the experiments is to show the effect batch normalization has on training and to also start exploring input distributions as seen in the paper How Batch Normalization Works.

### Running Experiments
* `cd bn`
* create a conda environment
  * for mac use `env-mac.yml`
  * for linux use `env-ubuntu.yml`
  * `conda create -f [environment].yml`
* Linear Model
    * can easily train on CPU
    * will train 3 models on MNIST
        * Linear with no BN
        * Linear with BN
        * Linear with BN and Noise
    * `python run_linear.py`
    * `tensorboard --logdir runs` to see results
* VGG Model
    * requires CPU to train
    * will train 3 models on CIFAR10
        * VGG11 with no BN
        * VGG11 with BN
        * VGG11 with BN and Noise
    * `python run_vgg.py`
    * `tensorboard --logdir runs` to see results