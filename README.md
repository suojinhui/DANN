## This is a pytorch implementation of the paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*

#### Network Structure


![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset

First, you need download the target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1pXaMkVsQf_yUT51SeYh27g) fetch code: kjan or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg)

```
cd dataset
tar -zvxf mnist_m.tar.gz
```

#### Training(suojinhui) 

```
CUDA_VISIBLE_DEVICES=1 python main.py # CUDA_VISIBLE_DEVICES=1 用于指定GPU编号，可选，不指定则默认使用0号GPU
```
### Change Log
- 2024-09-26: 重整代码，增加注释，增加可视化，增加训练参数，增加训练日志，重写梯度反转层。
- 2024-09-26: 合并test.py和main.py，简化训练流。