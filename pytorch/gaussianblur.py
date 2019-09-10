import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import numpy as np


def gaussian_kernel_1d(std: float, kernel_size: int) -> torch.FloatTensor:
    x = torch.FloatTensor([range(- kernel_size // 2 + 1, kernel_size // 2 + 1)])
    g = np.exp(- (x**2 / (2 * std**2)) / (np.sqrt(2 * np.pi) * std))
    # normalize the sum to 1
    g = g / g.sum()
    return g


def nearest_odd_integer(number, higher=True):
    integer = int(np.ceil(number))
    odd = integer if integer % 2 == 1 else integer+1
    return odd if higher else odd-2


def appropriate_kernel_size(std: float) -> int:
    """
    Returns the appropriate gaussian kernel size to be used for a given standard deviation.
    """
    # nearest odd number to 5*std.
    kernel_size = nearest_odd_integer(5 * std)
    # the kernel shouldn't be smaller than 3.
    return max(kernel_size, 3)


class GaussianBlur(nn.Module):
    def __init__(self, std = 2.0):
        super().__init__()
        self.std = nn.Parameter(torch.Tensor([std]), requires_grad=False)
        self._image_size = None

    @property
    def kernel(self):
        return gaussian_kernel_1d(std=self.std, kernel_size=self.kernel_size)

    @property
    def kernel_size(self) -> int:
        """
        Determines the kernel size dynamically depending on the std.
        We limit the kernel size to the smallest image dimension, at most.
        """
        k_size = appropriate_kernel_size(self.std)
        
        if self._image_size:
            # can't have kernel bigger than image size.
            max_k_size = nearest_odd_integer(self._image_size, higher=False)
            k_size = min(k_size, max_k_size)

        assert k_size % 2 == 1, "kernel size should be odd"
        return k_size

    def forward(self, inputs):
        #smallest image dimension 
        self._image_size = min(inputs.shape[-2:])

        k1 = self.kernel.view([1,1,1,-1])
        k2 = k1.transpose(-1, -2)

        kernel_size = k1.shape[-1]
        pad_amount = kernel_size // 2 #'same' padding.
        # Gaussian filter is separable:
        out_1 = F.conv2d(inputs, k1, padding=(0, pad_amount))
        out_2 = F.conv2d(out_1, k2, padding=(pad_amount, 0))
        return out_2




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

for step, (image_batch, label_batch) in enumerate(dataset):
    plt.imshow(image_batch[0, 0])
    image_batch.numpy()
    blurred = blur(image_batch)
    print(blurred.shape)
    plt.imshow(blurred[0, 0])
    plt.show()
    exit()