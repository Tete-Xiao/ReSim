# ReSim 



<p align="center">
  <img width="1331" alt="ReSim pipeline" src="https://user-images.githubusercontent.com/1455579/114447371-2f37ac00-9b87-11eb-8423-b0f896197136.png">
</p>

This repository provides the PyTorch implementation of Region Similarity Representation Learning (ReSim) described in [this paper](https://arxiv.org/abs/2103.12902): 

```
@Article{xiao2021region,
  author  = {Tete Xiao and Colorado J Reed and Xiaolong Wang and Kurt Keutzer and Trevor Darrell},
  title   = {Region Similarity Representation Learning},
  journal = {arXiv preprint arXiv:2103.12902},
  year    = {2021},
}
```

tldr; ReSim maintains spatial relationships in the convolutional feature maps when performing instance contrastive pre-training, which is useful for region-related tasks such as object detection, segmentation, and dense pose estimation.

## Installation
Assuming a conda environment:
```
conda create --name resim python=3.7
conda activate resim

# NOTE: if you are not using CUDA 10.2, you need to change the 10.2 in this command appropriately. 
# Code tested with torch 1.6 and 1.7
# (check CUDA version with e.g. `cat /usr/local/cuda/version.txt`)
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

## Pre-training

This codebase is based on the original [MoCo codebase](https://github.com/facebookresearch/moco) -- see this README for more details. 

To pre-train for 200 epochs using the ReSim-FPN implementation as described in the paper:

```
python main_moco.py -a resnet50 --lr 0.03 --batch-size 256 \
       --dist-url tcp://localhost:10005 --multiprocessing-distributed --world-size 1 --rank 0 \
       --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 \
       /location/of/imagenet/data/folder
```


## ResNet-50 Pre-trained Models

| Checkpoint | Pre-train Epochs | COCO AP @2x | MoCo Checkpoint                                                                 | Detectron Backbone                                                                       |
| ---------- | --------------- | ----------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| ReSim-FPN  | 400             | 41.9        | [Download](https://people.eecs.berkeley.edu/~cjrd/data/resim_fpn_400ep.pth.tar) | [Download](https://people.eecs.berkeley.edu/~cjrd/data/resim_fpn_backbone_400ep.pth.tar)                                                                                         |
| ReSim-FPN  | 200             | 41.4        | [Download](https://people.eecs.berkeley.edu/~cjrd/data/resim_fpn_200ep.pth.tar) | [Download](https://people.eecs.berkeley.edu/~cjrd/data/resim_fpn_backbone_200ep.pth.tar) |
| ReSim-C4   | 200             | 41.1        | [Download](https://people.eecs.berkeley.edu/~cjrd/data/resim_c4_200ep.pth.tar)  | [Download](https://people.eecs.berkeley.edu/~cjrd/data/resim_c4_backbone_200ep.pth.tar)  |


## Detection

See [these instructions](detection/README.md) for more details, but in brief:

```bash
# first install detectron2
# then place COCO-2017 dataset detection/datasets/coco

cd detection
python convert-pretrain-to-detectron2.py ../resim_fpn_checkpoint_latest.pth.tar detectron_resim_fpn_checkpoint_latest.pth.tar
python train_net.py --dist-url 'tcp://127.0.0.1:17654' --config-file configs/coco_R_50_FPN_2x_moco.yaml --num-gpus 8 MODEL.WEIGHTS detectron_resim_fpn_checkpoint_latest.pth.tar TEST.EVAL_PERIOD 180000 OUTPUT_DIR results/coco2x-resim-fpn SOLVER.CHECKPOINT_PERIOD 180000
```


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](./LICENSE).
