# Evaluation

This part is used to obtain the performance of all DNN models on each dataset. In order to ensure the fairness of evaluation, we introduce third-party projects weiaicunzai/pytorch-cifar100 publicly available on GitHub as our basic implementation and improve on basis of it. The evaluation results of CIFAR-10 are reused from aiboxlab/imagedataset2vec.

## Experiment eviroument


- python  3.9.2
- pytorch  1.13.0 
- torch  1.12.1 
- tensorboard 2.4.0


## Usage



You can specify the net to train using arg -net, the batchsize and dataset number you want during training. Different dataset numbers correspond to different image datasets. For datasets from CIFAR-100, the numbers range from 0 to 159.

```bash
# use gpu to train shufflenetv2
$ python3 train.py -net shufflenetv2 -b batchsize -d datasetnum -gpu 
```







Test the model
```bash
$ python3 test.py -net shufflenetv2 -weights path_to_shufflenetv2_weights_file
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- googlenet [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- shufflenetv2 [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- mobilenetv2 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)



