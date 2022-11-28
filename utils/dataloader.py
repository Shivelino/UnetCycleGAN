import glob
import random
import os

import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CycleGanDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        # PIL 默认rgb
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        # opencv type
        # item_A = self.transform(cv2.imread(self.files_A[index % len(self.files_A)]))  # bgr
        # if self.unaligned:
        #     img = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     item_B = self.transform(img)
        # else:
        #     img = cv2.imread(self.files_B[index % len(self.files_B)])
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     item_B = self.transform(img)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
