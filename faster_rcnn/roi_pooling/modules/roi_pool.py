from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction
from torch import Tensor


class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pooled_width = pooled_width
        self.pooled_height = pooled_height
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        return RoIPoolFunction.apply(features, rois, Tensor([int(self.pooled_height)]), Tensor([int(self.pooled_width)]), Tensor([float(self.spatial_scale)]))
