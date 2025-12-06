import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import struct


class FashionMNISTDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.images = self._load_image(image_path)
        self.labels = self._load_label(label_path)
        self.transform = transform

    def _load_image(self, path):
        with open(path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f'Invalid magic number for image file: {magic}')
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
        return images

    def _load_label(self, path):
        with open(path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f'Invalid magic number for label file: {magic}')
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def __len__(self):
        """
        Return the number of samples in the dataset.
        :return: int, the number of samples
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Select a sample by index.
        :param index: int, the index of the sample
        :return: image and label of the sample
        """
        image = self.images[index]
        label = self.labels[index]
        image = self.transform(image)

        return image, label


def fashion_mnist_data_loader(config, mode='train'):
    """

    :param config:
    :param mode: String, 'train' or 'test'
    :return: DataLoader, one batch of images and labels
    """
    image_path = os.path.join(config['data']['data_dir'], config['data'][f'{mode}_images_file'])
    label_path = os.path.join(config['data']['data_dir'], config['data'][f'{mode}_labels_file'])

    batch_size = config['model']['batch_size']

    if mode == 'train':
        transform = transforms.Compose([
            # Convert to PIL
            transforms.ToPILImage(),
            # Flips the image horizontally
            transforms.RandomHorizontalFlip(p=0.5),
            # Create random shifts of images
            transforms.RandomAffine(
                degrees=7.5,
                translate=(0.078, 0.075),
                scale=(0.915, 1.085)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset = FashionMNISTDataset(
            image_path,
            label_path,
            transform
        )
        train_size = int(0.9 * len(dataset))
        validate_size = len(dataset) - train_size
        train_set, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=True)
        return train_loader, validate_loader
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset = FashionMNISTDataset(
            image_path,
            label_path,
            transform
        )
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader
