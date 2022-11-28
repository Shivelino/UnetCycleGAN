# CycleGAN实验(包括Unet-CycleGAN)
<hr/>

## 环境配置
```shell
pip install -r requirements.txt
```

## 数据集
- 数据集格式参照```dataset/example```
- 可以使用CycleGAN等网络使用的pix2pix数据集
- 我们提供的flower41-49和flower51-98数据集可在Release发行版中找到

## 运行
- 训练
```shell
python train.py [--dataset] [--unet]
```
- 测试
```shell
python test.py [--dataset] [--unet]
```

## 可能的bug
### visdom报错
- 运行下面的指令手动启动visdom，并注释掉train.py中有"NOTEST"标记的注释行指向的两行判断逻辑
```shell
python -m visdom.server
```

### test模型报错
- 训练的模型和测试时指定的网络不一致

### dataset路径报错
- 尝试检查dataset有没有按照example的格式放置

