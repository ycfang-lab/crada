"""
@author: Jiahua Wu
@contact: jhwu@shu.edu.cn
"""
import random

from PIL import ImageFilter
class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like (h,w), output size will be
        matched to this. If size is an int, output will be (size, size)

    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x