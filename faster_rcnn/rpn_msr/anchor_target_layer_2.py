import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr


from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps

from ..fastrcnn.bbox_transform import bbox_transform
from ..config import cfg


class AnchorTargerLayer(nn.Module):

    def __init__(self, feat_stride, anchor_scales):
        super(AnchorTargerLayer, self).__init__()
        self._feat_stride = feat_stride
        self._anchor_scales = anchor_scales

        self._anchors = torch.from_numpy(
            generate_anchors(scales=np.array(self._anchor_scales)))

        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allow_border = 0

    def forward(self, input):
        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        return rpn_cls_score
