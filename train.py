import os
import numpy as np
import cv2 as cv
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import v2
import matplotlib.pyplot as plt

# parameters
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# result directory
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True ):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)

        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)

        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)

        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1, stride=1, padding=0, bias=True)
 
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.maxpool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.maxpool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.maxpool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.maxpool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


# xBD dataset
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

        # 만약 label이 [0, 1] 범위로 이미 정규화되어 있다면 아래 코드는 제거하거나 주석 처리합니다.
        # label = label / 255.0
        input = input / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)
        
        return data

# Transform
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
    Normalization(mean=0.5, std=0.5)
])

dataset_train = xBD(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

dataset_val = xBD(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

net = UNet().to(device)

# Loss function 설정하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_pred = lambda output: (torch.sigmoid(output) > 0.5).float()
fn_acc = lambda pred, label: (pred == label).float().mean()

## Tensorboard 를 사용하기 위한 summarywriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 로드하기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch  = 0
        return net, optim, epoch
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## 네트워크 학습
st_epoch = 0

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []
    acc_arr = []

    for batch, data in enumerate(loader_train, 1):
        # fastward pass
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        # backward pass
        optim.zero_grad()

        loss = fn_loss(output, label)
        loss.backward()

        optim.step()

        # loss function 계산
        loss_arr.append(loss.item())

        # 예측값 계산: sigmoid -> threshold 적용
        pred = fn_pred(output)
        acc = fn_acc(pred, label)
        acc_arr.append(acc.item())

        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr), np.mean(acc_arr)))
            
        # Tensorboard 저장하기
        label_np = fn_tonumpy(label)
        input_np = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output_np = fn_tonumpy(pred)

        writer_train.add_image('label', label_np, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
        writer_train.add_image('input', input_np, num_batch_train * (epoch -1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output_np, num_batch_train * (epoch -1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
    writer_train.add_scalar('accuracy', np.mean(acc_arr), epoch)

    with torch.no_grad():
        net.eval()
        loss_arr = []
        acc_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # loss function 계산
            loss = fn_loss(output, label)
            loss_arr.append(loss.item())

            # 예측값 계산: sigmoid -> threshold 적용
            pred = fn_pred(output)
            acc = fn_acc(pred, label)
            acc_arr.append(acc.item())

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
                  (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr), np.mean(acc_arr)))
            
            # Tensorboard 저장하기
            label_np = fn_tonumpy(label)
            input_np = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output_np = fn_tonumpy(pred)

            writer_val.add_image('label', label_np, num_batch_val * (epoch -1) + batch, dataformats='NHWC')
            writer_val.add_image('input', input_np, num_batch_val * (epoch -1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output_np, num_batch_val * (epoch -1) + batch, dataformats='NHWC')

    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    writer_val.add_scalar('accuracy', np.mean(acc_arr), epoch)

    if epoch % 5 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()