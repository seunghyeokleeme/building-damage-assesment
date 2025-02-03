import os
import numpy as np
import tarfile
from PIL import Image
from matplotlib import pyplot as plt

def extract_tar(tar_path, extract_to_path):
    """tar 압축풀기"""
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to_path)
    print(f"{extract_to_path} 경로에 압축을 해제합니다.")

# 데이터 읽어오는 함수
dir_data = './xbd'

# extract_tar(os.path.join(dir_data, 'train_images_labels_targets.tar'), dir_data)
# extract_tar(os.path.join(dir_data, 'test_images_labels_targets.tar'), dir_data)

lst_train_images = [f for f in os.listdir(os.path.join(dir_data, 'train', 'images')) if f.endswith('pre_disaster.png')]
lst_train_targets = [f for f in os.listdir(os.path.join(dir_data, 'train', 'targets')) if f.endswith('pre_disaster_target.png')]
lst_test_images = [f for f in os.listdir(os.path.join(dir_data, 'test', 'images')) if f.endswith('pre_disaster.png')]
lst_test_targets = [f for f in os.listdir(os.path.join(dir_data, 'test', 'targets')) if f.endswith('pre_disaster_target.png')]

lst_train_images.sort()
lst_train_targets.sort()
lst_test_images.sort()
lst_test_targets.sort()

print(f"Number of training images: {len(lst_train_images)}")
print(f"Number of training targets: {len(lst_train_targets)}")
print(f"Number of test images: {len(lst_test_images)}")
print(f"Number of test targets: {len(lst_test_targets)}")

print(lst_train_images[0])
print(lst_train_targets[0])
print(lst_test_images[0])
print(lst_test_targets[0])

dir_save_data = './datasets'

# lst_train_images: n_train + n_val
n_train = 2239
n_val = 560
n_test = 933

dir_save_train = os.path.join(dir_save_data, 'train')
dir_save_val = os.path.join(dir_save_data, 'val')
dir_save_test = os.path.join(dir_save_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

id_frame = np.arange(n_train+n_val)
np.random.shuffle(id_frame)
offset = 0

for i in range(n_train):
    idx = id_frame[i+offset]
    fname_image = lst_train_images[idx].split('.png')[0]
    fname_label = lst_train_targets[idx].split('_target.png')[0]
    path_image = os.path.join(dir_data, 'train', 'images', lst_train_images[idx])
    path_label = os.path.join(dir_data, 'train', 'targets', lst_train_targets[idx])    
    
    img_input = Image.open(path_image)
    img_label = Image.open(path_label)

    input_ = np.asarray(img_input)
    label_ = np.asarray(img_label)

    np.save(os.path.join(dir_save_train, '%s_input.npy' %fname_image), input_)
    np.save(os.path.join(dir_save_train, '%s_label.npy' %fname_label), label_)

print("Data are ready to be used for training.")

offset += n_train

for i in range(n_val):
    idx = id_frame[i+offset]
    fname_image = lst_train_images[idx].split('.png')[0]
    fname_label = lst_train_targets[idx].split('_target.png')[0]
    path_image = os.path.join(dir_data, 'train', 'images', lst_train_images[idx])
    path_label = os.path.join(dir_data, 'train', 'targets', lst_train_targets[idx])    
    
    img_input = Image.open(path_image)
    img_label = Image.open(path_label)

    input_ = np.asarray(img_input)
    label_ = np.asarray(img_label)

    np.save(os.path.join(dir_save_val, '%s_input.npy' %fname_image), input_)
    np.save(os.path.join(dir_save_val, '%s_label.npy' %fname_label), label_)

print("Data are ready to be used for validation.")

id_frame = np.arange(n_test)
np.random.shuffle(id_frame)
offset = 0

for i in range(n_test):
    idx = id_frame[i+offset]
    fname_image = lst_test_images[idx].split('.png')[0]
    fname_label = lst_test_targets[idx].split('_target.png')[0]
    path_image = os.path.join(dir_data, 'test', 'images', lst_test_images[idx])
    path_label = os.path.join(dir_data, 'test', 'targets', lst_test_targets[idx])

    img_input = Image.open(path_image)
    img_label = Image.open(path_label)

    input_ = np.asarray(img_input)
    label_ = np.asarray(img_label)

    np.save(os.path.join(dir_save_test, '%s_input.npy' %fname_image), input_)
    np.save(os.path.join(dir_save_test, '%s_label.npy' %fname_label), label_)

print("Data are ready to be used for testing.")

print(n_train *2 , len(os.listdir(dir_save_train)))
print(n_val * 2, len(os.listdir(dir_save_val)))
print(n_test * 2, len(os.listdir(dir_save_test)))

print("Data are successfully saved.")
print("Data preprocessing is done.")

print(input_.shape, label_.shape)

plt.subplot(121)
plt.imshow(input_)
plt.title('Input Image')

plt.subplot(122)
plt.imshow(label_, cmap='gray')
plt.title('Label Image')

plt.show()