import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import numpy as np

from gaussianblur import GaussianBlur

data_dir = "./data"
batch_size = 32
mnist_transforms = [transforms.ToTensor()]
use_cuda = torch.cuda.is_available()
print("use cuda:", use_cuda)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
dataset = torch.utils.data.DataLoader(
    datasets.MNIST(
        data_dir,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    ), 
    batch_size=32, **kwargs
)

blur = GaussianBlur()

import matplotlib.pyplot as plt


if __name__ == "__main__":
    for step, (image_batch, label_batch) in enumerate(dataset):
        plt.imshow(image_batch[0, 0])
        image_batch.numpy()
        blurred = blur(image_batch)
        print(blurred.shape)
        plt.imshow(blurred[0, 0])
        plt.show()
        exit()