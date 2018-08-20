from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction
from torch import Tensor


class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pooled_width = Tensor([int(pooled_width)])
        self.pooled_height = Tensor([int(pooled_height)])
        self.spatial_scale = Tensor([float(spatial_scale)])

    def forward(self, features, rois):
        return RoIPoolFunction.apply(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)
