import pickle as pkl
import os
import os.path as op
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


def res_grid(ori_folder, res_folder, save_folder):
    """拼接原图和效果图"""
    ori_img_names = os.listdir(ori_folder)
    res_img_names = os.listdir(res_folder)
    assert len(ori_img_names) == len(res_img_names)
    trans = transforms.Compose([transforms.ToTensor()])
    for ind in tqdm(range(len(res_img_names))):
        ori_img = cv2.cvtColor(cv2.imread(op.join(ori_folder, ori_img_names[ind])), cv2.COLOR_BGR2RGB)
        res_img = cv2.cvtColor(cv2.imread(op.join(res_folder, res_img_names[ind])), cv2.COLOR_BGR2RGB)

        ax1 = plt.subplot(121)  # 左边原图
        plt.axis('off')  # 关坐标轴
        plt.imshow(ori_img)

        ax2 = plt.subplot(122)  # 右边原图
        plt.axis('off')
        plt.imshow(res_img)

        plt.tight_layout()
        plt.savefig(op.join(save_folder, "{}.jpg".format(ind)), bbox_inches='tight')
        # plt.show()


def res_grid_epoches(ori_folder, __epoch__, save_epoch_folder):
    """拼接不同epoch的效果为grid"""
    ori_img_names = os.listdir(ori_folder)
    for ind in tqdm(range(len(ori_img_names))):
        ori_img = cv2.cvtColor(cv2.imread(op.join(ori_folder, ori_img_names[ind])), cv2.COLOR_BGR2RGB)
        ax1 = plt.subplot(161)  # 左边原图
        plt.axis('off')  # 关坐标轴
        plt.imshow(ori_img)
        for i in range(0, 5):
            res_folder = res_folders.replace("__epoch__", str(i * 50))
            res_img_names = os.listdir(res_folder)
            assert len(ori_img_names) == len(res_img_names)
            res_img = cv2.cvtColor(cv2.imread(op.join(res_folder, res_img_names[ind])), cv2.COLOR_BGR2RGB)

            ax2 = plt.subplot(162 + i)  # 右边原图
            plt.axis('off')
            plt.imshow(res_img)

        plt.tight_layout()
        plt.savefig(op.join(save_epoch_folder, "{}.jpg".format(ind)), bbox_inches='tight')
        # plt.show()


def stack_imgs(img_name_folder, img_names):
    """堆叠图像"""
    pass


if __name__ == '__main__':
    # tag = "fl41_49"
    tag = "fl51_98"
    # unet_tag = ""
    unet_tag = "_unet"
    # form_direct = "A2B"
    form_direct = "B2A"

    long_tag = None
    if tag == "fl41_49" or tag == "fl41_49_unet":
        long_tag = "flower41toflower49"
    elif tag == "fl51_98" or tag == "fl51_98_unet":
        long_tag = "flower51toflower98"
    else:
        raise RuntimeError("Tag Error")
    ori_folder = r"E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\CycleGAN\datasets\{}\test\{}".format(
        long_tag, form_direct[0])
    res_folders = r"E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\CycleGAN\output\{}{}\__epoch__\{}".format(
        tag,
        unet_tag,
        form_direct[
            -1])
    res_folder = r"E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\CycleGAN\output\{}{}\200\{}".format(tag,
                                                                                                                 unet_tag,
                                                                                                                 form_direct[
                                                                                                                     -1])
    save_folder = r"E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\CycleGAN\cat_imgs\{}{}\{}".format(tag,
                                                                                                                unet_tag,
                                                                                                                form_direct)
    save_epoch_folder = r"E:\Study\_2022_fall\deeplearn_cv\BigHomework\BigHomework2\CycleGAN\cat_imgs_epoch\{}{}\{}".format(
        tag,
        unet_tag,
        form_direct)
    # res_grid(ori_folder, res_folder, save_folder)
    res_grid_epoches(ori_folder, res_folders, save_epoch_folder)
