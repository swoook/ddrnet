# ❗ This is cloned repository!

This repository is cloned from [chenjun2hao/DDRNet.pytorch](https://github.com/chenjun2hao/DDRNet.pytorch) and modified for research

# Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes

## Introduction
This is the unofficial code of [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/pdf/2101.06085.pdf). which achieve state-of-the-art trade-off between accuracy and speed on cityscapes and camvid, without using inference acceleration and extra data!on single 2080Ti GPU, DDRNet-23-slim yields 77.4% mIoU at 109 FPS on Cityscapes test set and 74.4% mIoU at 230 FPS on CamVid test set.

The code mainly borrows from [HRNet-Semantic-Segmentation OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR) and [the official repository](https://github.com/ydhongHIT/DDRNet), thanks for their work.


<!-- ![](figures/ddrnet.png) -->
<figure>
  <text-align: center;>
  <center>
  <img src="./figures/ddrnet.png" alt="hrnet" title="" width="400" height="400" />
  </center>
</figcaption>
</figure>
<sup>A  comparison  of  speed-accuracy  trade-off  on  Cityscapes  test  set.</sup>

## Requirements

```
torch>=1.7.0
cudatoolkit>=10.2
```

## Cityscapes Data Preparation

1. Download two files below from [Cityscapes](https://www.cityscapes-dataset.com/).to the *\${CITYSCAPES_ROOT}*

   * *leftImg8bit_trainvaltest.zip*
   * *gtFine_trainvaltest.zip*

2. Unzip them

3. Rename the folders like below

   ```
   └── cityscapes
     ├── leftImg8bit
         ├── test
         ├── train
         └── val
     └── gtFine
         ├── test
         ├── train
         └── val
   ```

4. Update some properties in *{REPO_ROOT}/experiments/cityscapes/${MODEL_YAML}* like below

   ```yaml
   DATASET:
     DATASET: cityscapes
     ROOT: ${CITYSCAPES_ROOT}
     TEST_SET: 'cityscapes/list/test.lst'
     TRAIN_SET: 'cityscapes/list/train.lst'
     ...
   ```

## Pretrained Models

* [The official repository](https://github.com/ydhongHIT/DDRNet) provides pretrained models for *Cityscapes* 

1. Download the pretrained model to *\${MODEL_DIR}*

2. Update `MODEL.PRETRAINED` and `TEST.MODEL_FILE` in *{REPO_ROOT}/experiments/cityscapes/${MODEL_YAML}* like below

   ```yaml
   ...
   MODEL:
     ...
     PRETRAINED: "${MODEL_DIR}/${MODEL_NAME}.pth"
     ALIGN_CORNERS: false
     ...
   TEST:
     ...
     MODEL_FILE: "${MODEL_DIR}/${MODEL_NAME}.pth"
     ...
   ```


## Validation

* Execute the command below to evaluate the model on *Cityscapes-val*

```
cd ${REPO_ROOT}
python tools/eval.py --cfg experiments/cityscapes/ddrnet23_slim.yaml
```

| model | OHEM | Multi-scale| Flip | mIoU | FPS | E2E Latency (s) | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | ---- | ---- |
| DDRNet23_slim | Yes | No | No | 77.83 | 91.31 | 0.062 | [official](https://github.com/ydhongHIT/DDRNet) |
| DDRNet23_slim | Yes | No | Yes| 78.42 | TBD | TBD | [official](https://github.com/ydhongHIT/DDRNet) |
| DDRNet23      | Yes | No | No | 79.51 | TBD | TBD | [official](https://github.com/ydhongHIT/DDRNet) |
| DDRNet23      | Yes | No | Yes| 79.98 | TBD | TBD | [official](https://github.com/ydhongHIT/DDRNet) |

**mIoU** denotes an mIoU on Cityscapes validation set.

**FPS** is measured by following the test code provided by SwiftNet. Refer to `speed_test` from [lib/utils/utils.py](lib/utils/utils.py) for more details.

**E2E Latency** denotes an end-to-end latency including pre/post-processing.

**Note**

- with the `ALIGN_CORNERS: false` in `***.yaml` will reach higher accuracy.


## TRAIN

download [the imagenet pretrained model](https://github.com/ydhongHIT/DDRNet), and then train the model with 2 nvidia-3080

```python
cd ${PROJECT}
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23_slim.yaml
```

**the own trained model coming soon**

## OWN model
| model | Train Set | Test Set | OHEM | Multi-scale| Flip | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| DDRNet23_slim | train | eval | Yes | No | Yes | 77.77 | [Baidu/password:it2s](https://pan.baidu.com/s/17pOOTc-HBG6TNf4k_cn4VA) |
| DDRNet23_slim | train | eval | Yes | Yes| Yes | 79.57 | [Baidu/password:it2s](https://pan.baidu.com/s/17pOOTc-HBG6TNf4k_cn4VA) |
| DDRNet23      | train | eval | Yes | No | Yes | ~ | None |
| DDRNet39      | train | eval | Yes | No | Yes | ~ | None |

**Note**
- set the `ALIGN_CORNERS: true` in `***.yaml`, because i use the default setting in [HRNet-Semantic-Segmentation OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).
- Multi-scale with scales: 0.5,0.75,1.0,1.25,1.5,1.75. it runs too slow.
- from [ydhongHIT](https://github.com/ydhongHIT), can change the `align_corners=True` with better performance, the default option is `False`

## Reference
[1] [HRNet-Semantic-Segmentation OCR branch](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR) 

[2] [the official repository](https://github.com/ydhongHIT/DDRNet)

