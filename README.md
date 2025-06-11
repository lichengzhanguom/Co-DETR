## Introduction
This repository is a fork of [Co-DETR](https://github.com/Sense-X/Co-DETR), which implements the Co-DETR model—a novel approach to DETR-based object detection introduced in the paper "DETRs with Collaborative Hybrid Assignments Training". We leverage Co-DETR for our task.

We employ Co-DETR with ViT-L, which is the best model on COCO benchmark (**the first model to achieve 66.0 AP on COCO test-dev**).

## Running

### Install
MMCV==1.7.0

mmdet==2.25.3

python=3.7.16

pytorch=1.13.0+cu117

timm==0.4.12

### Data
Transfer the dataset to COCO format. The dataset should be organized as:
```
Co-DETR
└── data
    └── coco
        ├── annotations
        │      ├── instances_train2017.json
        │      └── instances_val2017.json
        ├── train2017
        └── val2017
      
```

### Training
Train Co-Deformable-DETR + ResNet-50 with 8 GPUs:
```shell
sh tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 8 path_to_exp
```
Train using slurm:
```shell
sh tools/slurm_train.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_exp
```

### Testing
Test Co-Deformable-DETR + ResNet-50 with 8 GPUs, and evaluate:
```shell
sh tools/dist_test.sh  projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint 8 --eval bbox
```
Test using slurm:
```shell
sh tools/slurm_test.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint --eval bbox
```

## Cite Co-DETR

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@inproceedings{zong2023detrs,
  title={Detrs with collaborative hybrid assignments training},
  author={Zong, Zhuofan and Song, Guanglu and Liu, Yu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6748--6758},
  year={2023}
}
```

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
