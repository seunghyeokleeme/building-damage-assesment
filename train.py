import os
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn

from torchvision.transforms import v2
import matplotlib.pyplot as plt

# 파라미터 설정
lr = 1e-3
batch_size = 4
num_epoch = 10

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 구축
class xBD(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.endswith('_label.npy')]
        lst_input = [f for f in lst_data if f.endswith('_input.npy')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input
    
    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)
        
        return data

# 트랜스폼 구현
class Resize(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        label, input = data['label'], data['input']

        input = cv.resize(input, self.shape[::-1], interpolation=cv.INTER_LINEAR)
        label = cv.resize(label, self.shape[::-1], interpolation=cv.INTER_NEAREST)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        data = {'label': label, 'input': input}

        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomHorizontalFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # Horizontal Flip
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        # if np.random.rand() > 0.5:
        #     label = np.flipud(label)
        #     input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

transform = v2.Compose([
    # Resize((512, 512)),
    RandomHorizontalFlip(),
    ToTensor(),
])

dataset_train = xBD(data_dir=os.path.join(data_dir, 'train'), transform=transform)
# dataset_train = xBD(data_dir=os.path.join(data_dir, 'train'), transform=None)

first_item = dataset_train.__getitem__(0)

f_label, f_input = first_item['label'], first_item['input']

f_label = f_label.numpy() * 255.0
f_input = f_input.numpy() * 255.0

f_label = f_label.transpose((1, 2, 0)).astype(np.uint8)
f_input = f_input.transpose((1, 2, 0)).astype(np.uint8)

plt.subplot(121)
plt.imshow(f_label, cmap='gray')
plt.title('Label')

plt.subplot(122)
plt.imshow(f_input)
plt.title('Input')

plt.show()