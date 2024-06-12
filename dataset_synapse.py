import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image

def combine_labels(fda_approved, ct_tox):
    if fda_approved == 0 and ct_tox == 0:
        return 0
    elif fda_approved == 1 and ct_tox == 0:
        return 1
    elif fda_approved == 0 and ct_tox == 1:
        return 2
    elif fda_approved == 1 and ct_tox == 1:
        return 3
    # 可以根据实际情况继续添加更多组合情况

def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image

def random_rotate(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image

class RandomGenerator(object):
    def __init__(self, output_size, device):
        self.output_size = output_size
        self.device = device

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 图像处理逻辑
        if random.random() > 0.5:
            image = random_rot_flip(image)
        elif random.random() > 0.5:
            image = random_rotate(image)

        x, y = image.shape[:2]  # 获取图像的高度和宽度

        if x != self.output_size[0] or y != self.output_size[1]:
            zoom_factor = (self.output_size[0] / x, self.output_size[1] / y)
            if len(image.shape) == 3:
                zoom_factor = zoom_factor + (1,)  # 添加通道维度
            image = zoom(image, zoom_factor, order=3)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(self.device)  # 转换为PyTorch张量并移动到指定设备

        sample = {'image': image, 'label': torch.tensor(label).to(self.device)}
        return sample

class MoleculeDataset(Dataset):
    def __init__(self, image_dir, image_list_file, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = self.load_image_files(image_list_file)
        self.labels = self.load_labels(label_file)

    def load_image_files(self, image_list_file):
        with open(image_list_file, 'r') as f:
            image_files = [line.strip() for line in f]
        return image_files

    def load_labels(self, label_file):
        labels = {}
        with open(label_file, 'r') as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                img_name = parts[0] + '.png'  # 假设图像文件名是SMILES字符串加上.png后缀
                fda_approved = int(parts[1])
                ct_tox = int(parts[2])
                combined_label = combine_labels(fda_approved, ct_tox)
                labels[img_name] = combined_label
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels.get(img_name, 0)  # 如果找不到标签，返回默认值
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], torch.tensor(sample['label']).clone().detach()