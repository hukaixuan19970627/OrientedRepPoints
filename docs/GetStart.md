1. 安装环境： 
    $ cd /.../OrientedRepPoints/DOTA_devkit
install swig
    sudo apt-get install swig
create the c++ extension for python
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace

    $ cd /.../OrientedRepPoints
    $ pip install -r requirements.txt
    $ python setup.py develop

2. 训练custom dataset
Dataset json file ready:
    $ python DOTA_devkit/DOTA2COCO.py
Start Demo
    $ CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --launcher pytorch \
>      --config 'configs/dota/swin_tiny_patch4_window7_dotav2.py'
    '''
    -------------------------------------------------------------------------------
    在终端执行程序时指定GPU ：CUDA_VISIBLE_DEVICES=1   python  your_file.py：
    CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
    CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
    CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
    CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; device 1 is masked
    CUDA_VISIBLE_DEVICES=""          No GPU will be visible
    -------------------------------------------------------------------------------
    在Python代码中指定GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''

How to use Swin Backbone in torch≤1.4：
    $ python tools/TorchModel_Save3toSave2.py
    
    