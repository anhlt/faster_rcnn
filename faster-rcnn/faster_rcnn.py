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