# CycleGAN Experiments(including Unet-CycleGAN)
<hr/>

## Environment
```shell
pip install -r requirements.txt
```

## Dataset
- Dataset format reference ```dataset/example```
- Pix2pix datasets used by networks such as CycleGAN can be used
- The datasets of flower41-49 and flower51-98 provided by us can be found in the [release](https://gitee.com/shivelino/UnetCycleGAN)

## Run
- Train
```shell
python train.py [--dataset] [--unet]
```
- Test
```shell
python test.py [--dataset] [--unet]
```

## Possible Bugs
### visdom error
- Run the following command to manually start visdom, and comment out the two lines of judgment logic pointed to by the comment line marked "NOTEST" in train.py.
```shell
python -m visdom.server
```

### model error of test.py
- The training model is inconsistent with the network specified in the test.

### dataset path error
- Check whether the dataset is placed in the format of "example".

