import random
from PIL import Image
import os
import os.path
import cv2
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        img1_folder = 'A'
        img2_folder = 'B'
        label_folder = 'label'

        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            basepath = data_root + "/test"
        elif split == 'val':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            basepath = data_root + "/val"
        elif split == 'train':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            basepath = data_root + "/train"

        image1_name = os.path.join(basepath, img1_folder, split + '_' + line_split[0] + '.png')
        image2_name = os.path.join(basepath, img2_folder, split + '_' + line_split[0] + '.png')
        label_name = os.path.join(basepath, label_folder, split + '_' + line_split[0] + '.png')
        item = (image1_name, image2_name, label_name, line_split[0])
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


def __crop(img, pos, size):
    ow, oh = img.size
    tw = th = size
    #
    if (ow > tw and oh > th):
        x, y = pos
        return img.crop((x, y, x + tw, y + th))
    else:
        return img  #


def get_transform(convert=True, normalize=False, crop=True, pos=None, size=256):
    transform_list = []
    if crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, pos, size)))  # 注意修改 256 128
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class SemData(Dataset):
    def __init__(self, config, dataset='LEVIR-CD', split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.config = config
        self.image_transform = None
        self.label_transform = None
        self.dataset = dataset

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.split == 'train':
            image1_path, image2_path, label_path, name = self.data_list[index]
        else:
            image1_path, image2_path, label_path, name = self.data_list[index]

        ##
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        h, w = image1.size
        size = 256  #
        if h > size and w > size:  #
            x = random.randint(0, h - size)
            y = random.randint(0, w - size)
            pos = [x, y]
        else:
            pos = [0, 0]
        self.image_transform = get_transform(convert=True, normalize=True, pos=pos, size=size)
        self.label_transform = get_transform(convert=True, pos=pos, size=size)
        image1 = self.image_transform(image1)
        image2 = self.image_transform(image2)
        label = self.label_transform(label)

        if self.split == 'train':
            return image1, image2, label, name, pos
        elif self.split == 'val':
            return image1, image2, label, name
        else:
            return image1, image2, label, name

    def _get_label(self, label):

        origin_label = np.copy(label)
        binary_label = origin_label / 255.0

        return binary_label

    def save_pred(self, args, preds, label, sv_path, name, suffix, split='test'):
        preds = preds * 255.0
        label = label * 255.0

        for i in range(preds.shape[0]):
            _pred = np.asarray(preds[i], dtype=np.uint8)
            cv2.imwrite((os.path.join(sv_path, split + '_' + name[i] + suffix)), _pred)
