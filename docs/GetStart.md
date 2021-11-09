# Getting Started

This page provides basic usage based MMdetection. For installation instructions, please see [install.md](install.md)

# Train a model

MMDetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

1. Prepare custom dataset files.
```shell
python DOTA_devkit/ImgSplit_multi_process.py
python DOTA_devkit/DOTA2COCO.py
```

2.1 Train  with a single GPU 

```shell
python tools/train.py --config 'configs/dota/r50_dota_demo.py'
```

2.2 Train with multiple(4) GPUs

```shell
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --launcher pytorch \
    --config 'configs/dota/r50_dota_demo.py'
```

2.3 Train with specified GPUs. (for example with GPU=2,3)

```shell
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --launcher pytorch \
    --config 'configs/dota/r50_dota_demo.py'
```
or add code:
```
import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

# Inferenece with pretrained models
We provide the testing scripts to evaluate the trained model.

Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/r50_dotav1/`.

1. Test OrientedRepPoints with single GPU.

```shell
python tools/test.py \
    --config 'configs/dota/r50_dotav1.py' \
    --checkpoint 'work_dirs/r50_dotav1/epoch_40.pth' \
    --out 'work_dirs/r50_dotav1/results.pkl'
```

2. Test OrientedRepPoints with 4 GPUs.
```shell
python -m torch.distributed.launch --nproc_per_node 4 tools/test.py --launcher pytorch \
    --config 'configs/dota/r50_dotav1.py' \
    --checkpoint 'work_dirs/r50_dotav1/epoch_40.pth' \
    --out 'work_dirs/r50_dotav1/results.pkl'
```

3. Parse the results.
```shell
python tools/parse_results_pkl/parse_dota_evaluation.py \
    --detection_pkl_path 'work_dirs/r50_dotav1/results.pkl' \
    --val_json 'data/dataset_demo_split/test_datasetdemo.json' \
    --outpath 'work_dirs/r50_dotav1/Task1_results'
``` 

4. Merge the results.
```shell
python DOTA_devkit/ResultMerge_multi_process.py \
    --scrpath 'work_dirs/r50_dotav1/Task1_results' \
    --dstpath 'work_dirs/r50_dotav1/Task1_results_merged'
```

5. Evaluate the results.
```shell
python DOTA_devkit/dota_evaluation_task1.py \
    --detpath 'work_dirs/r50_dotav1/Task1_results_merged/Task1_{:s}.txt' \
    --annopath 'data/dataset_demo/labelTxt/{:s}.txt' \
    --imagesetfile 'data/dataset_demo/imgnamefile_demo.txt'
```

6. Visualize the results.
```shell
python tools/parse_results_pkl/show_learning_points_and_boxes.py
```

*If you want to evaluate the result on DOTA test-dev, please read the results.pkl, and run mergs the txt results. and zip the files  and submit it to the  [evaluation server](https://captain-whu.github.io/DOTA/index.html).
