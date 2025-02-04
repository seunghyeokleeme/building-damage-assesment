import os
from train import xBD
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data_dir = './datasets'
result_dir = './results'

dataset_train = xBD(data_dir=os.path.join(data_dir, 'train'), transform=None)
first_item = dataset_train.__getitem__(0)

f_label, f_input = first_item['label'], first_item['input']

print(f_label.shape, f_input.shape)

x = f_label.shape[0]
y = f_label.shape[1]

label_512 = f_label[:x//2, :y//2, :]
input_512 = f_input[:x//2, :y//2, :]

print(label_512.shape, input_512.shape)

f_input = f_input * 255.0
input_512 = input_512 * 255.0

f_label = f_label.astype(np.uint8)
f_input = f_input.astype(np.uint8)
label_512 = label_512.astype(np.uint8)
input_512 = input_512.astype(np.uint8)

plt.imsave(os.path.join(result_dir, 'png', 'label.png'), f_label.squeeze().astype(np.uint8), cmap='gray')
plt.imsave(os.path.join(result_dir, 'png', 'input.png'), f_input)
plt.imsave(os.path.join(result_dir, 'png', 'label_512.png'), label_512.squeeze().astype(np.uint8), cmap='gray')
plt.imsave(os.path.join(result_dir, 'png', 'input_512.png'), input_512)

label_image = Image.open(os.path.join(result_dir, 'png', 'label.png'))

label_image = np.asarray(label_image)

print(label_image.shape)
print(np.unique(label_image))

plt.subplot(141)
plt.imshow(f_label, cmap='gray')
plt.title('Label')

plt.subplot(142)
plt.imshow(f_input)
plt.title('Input')

plt.subplot(143)
plt.imshow(label_512, cmap='gray')
plt.title('Label 512')

plt.subplot(144)
plt.imshow(input_512)
plt.title('Input 512')

plt.show()
