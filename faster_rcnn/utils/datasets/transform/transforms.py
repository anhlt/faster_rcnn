from __future__ import division
import random
import numpy as np
from PIL import Image
from . import functional as F


__all__ = ["Compose", "RandomHorizontalFlip", "Resize"]


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        resized_img = F.resize(img, self.size, self.interpolation)
        ratio = min(resized_img.size) / min(img.size)

        return resized_img, bboxes * ratio

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
            img_center = np.array(img.size) / 2
            img_center = np.hstack((img_center, img_center)).astype(np.int32)
            if random.random() < self.p:
                img =  F.hflip(img)
                bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])

                box_w = abs(bboxes[:,0] - bboxes[:,2])

                bboxes[:,0] -= box_w
                bboxes[:,2] += box_w

            return img, bboxes

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
