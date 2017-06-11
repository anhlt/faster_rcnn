import cv2
import numpy as numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob

from lib.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from lib.bbox_transform import bbox_transform_inv, clip_boxes
from network import vgg16, Conv2d


class RPN(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()
        self.features = vgg16()
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(
            512, len(self.anchor_scales) * 3 * 2, 1, relu=False)
        self.bbox_conv = Conv2d(
            512, len(self.anchor_scales) * 3 * 4, 1, relu=False)

        self.cross_entropy = None
        self.loss_box = None

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10
