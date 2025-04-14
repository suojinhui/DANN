# DANN-amp

## This is a pytorch implementation of the paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

#### Network Structure


![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset

First, you need download the target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1pXaMkVsQf_yUT51SeYh27g) fetch code: kjan or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg)

```bash
cd dataset
tar -zvxf mnist_m.tar.gz
```

#### Training(suojinhui) 

- Specify GPU for training

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

- View optional parameters

```bash
python main.py --help 
```

- Train using the run.sh script (some parameters are already specified)

```bash
CUDA_VISIBLE_DEVICES=1 run.sh --data_root ./dataset --model_dir ./runs
```

### Change Log
- 2024-09-26: Refactor the code, add comments, add visualization, add training parameters, add training logs, rewrite the gradient reversal layer.
- 2024-09-26: Merge test.py and main.py, simplify the training flow.
- 2025-04-15: Add amp mixed precision training, optimize the training flow, use parameter table for passing parameters.