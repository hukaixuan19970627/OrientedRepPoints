# INSTAllation 
## Requirements
* Linux
* Python 3.7+ 
* PyTorch ≤ 1.4 (We haven't tested higher version)
* CUDA 9.0 or higher
* mmdet==1.1.0
* ![mmcv](https://github.com/open-mmlab/mmcv)==0.6.2
* GCC 4.9 or higher
* NCCL 2

We have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04
* CUDA: 10.0/10.1
* NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
* GCC(G++): 4.9/5.3/5.4/7.3

## Install 
**a. Create a conda virtual environment and activate it.**  
```
conda create -n orientedreppoints python=3.8 -y 
source activate orientedreppoints
```
**b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 10.1 ≤ 10.2)**
```
nvcc -V
nvidia-smi
```
**c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,**
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```
**d. Clone the orientedreppoints_dota repository.**
```
git clone https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA.git
cd OrientedRepPoints_DOTA
```
**e. Install orientedreppoints_dota.**

```python 
pip install -r requirements.txt
python setup.py develop  #or "pip install -v -e ."
```

## Install DOTA_devkit

```
cd OrientedRepPoints_DOTA/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## Prepare dataset
It is recommended to symlink the dataset root to $orientedreppoints/data. If your folder structure is different, you may need to change the corresponding paths in config files.
```
orientedreppoints
|——mmdet
|——tools
|——configs
|——data
|  |——dota
|  |  |——trainval_split
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——trainval.json
|  |  |——test_split
|  |  |  |——images
|  |  |  |——test.json
|  |——HRSC2016(OPTINAL)
|  |  |——Train
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——train.txt
|  |  |  |——trainval.json
|  |  |——Test
|  |  |  |——images
|  |  |  |——test.txt
|  |  |  |——test.json
|  |——UCASAOD(OPTINAL)
|  |  |——Train
|  |  |  |——images
|  |  |  |——labelTxt
|  |  |  |——train.txt
|  |  |  |——trainval.json
|  |  |——Test
|  |  |  |——images
|  |  |  |——test.txt
|  |  |  |——test.json
```
Note:
* `train.txt` and `test.txt` in HRSC2016 and UCASAOD are `.txt` files recording image names without extension.
* Without the pre-divided `train`，`test`, and `val` sub-dataset, the partition of UCASAOD dataset follows the [rep](https://github.com/ming71/UCAS-AOD-benchmark).
