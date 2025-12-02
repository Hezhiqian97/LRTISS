import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import cvtColor, preprocess_input

class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train=True, dataset_path=""):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        name = self.annotation_lines[index].split()[0]
        img_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg")
        label_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        # 读取图像
        image = cv2.imread(img_path)[:,:,::-1]  # BGR -> RGB
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 数据增强
        image, label = self._augment(image, label, self.input_shape, self.train)

        # 图像归一化并转通道
        image = preprocess_input(image.astype(np.float32))
        image = np.transpose(image, (2,0,1))

        # 处理超出类别的标签
        label[label >= self.num_classes] = self.num_classes

        # one-hot encoding，可延迟到 GPU 上
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[label.reshape(-1)]
        seg_labels = seg_labels.reshape((*self.input_shape, self.num_classes + 1))

        return image, label, seg_labels

    def _rand(self, a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def _augment(self, image, label, target_shape, randomize=True, jitter=0.3, hue=0.1, sat=0.7, val=0.3):
        """安全的高效数据增强"""
        h, w = target_shape
        ih, iw = image.shape[:2]

        if not randomize:
            scale = min(w/iw, h/ih)
            nw, nh = int(iw*scale), int(ih*scale)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

            new_image = np.full((h, w, 3), 128, dtype=np.uint8)
            new_label = np.zeros((h, w), dtype=np.uint8)
            dx, dy = (w - nw)//2, (h - nh)//2
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            new_label[dy:dy+nh, dx:dx+nw] = label
            return new_image, new_label

        # 随机缩放 + 扭曲
        new_ar = iw/ih * self._rand(1-jitter,1+jitter)/self._rand(1-jitter,1+jitter)
        scale = self._rand(0.25,2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)

        # 防止缩放超过目标尺寸
        nw = min(nw, w)
        nh = min(nh, h)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # 随机翻转
        if self._rand() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        # 灰条填充
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        new_label = np.zeros((h, w), dtype=np.uint8)
        dx = np.random.randint(0, w - nw + 1)
        dy = np.random.randint(0, h - nh + 1)
        # 防止越界
        new_image[dy:dy+nh, dx:dx+nw, :] = image[:nh, :nw, :]
        new_label[dy:dy+nh, dx:dx+nw] = label[:nh, :nw]

        # HSV 色域变换
        r = np.random.uniform(-1, 1, 3) * np.array([hue, sat, val]) + 1
        hsv = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return new_image, new_label


def unet_dataset_collate(batch):
    images, labels, seg_labels = zip(*batch)
    images = torch.from_numpy(np.array(images, dtype=np.float32))
    labels = torch.from_numpy(np.array(labels, dtype=np.int64))
    seg_labels = torch.from_numpy(np.array(seg_labels, dtype=np.float32))
    return images, labels, seg_labels
