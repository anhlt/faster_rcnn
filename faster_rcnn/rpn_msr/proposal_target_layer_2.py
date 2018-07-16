import torch.nn as nn
import numpy as np
from .generate_anchors import generate_anchors
from ..fastrcnn.bbox_transform import bbox_transform_inv, clip_boxes
from ..fastrcnn.nms_wrapper import nms
from ..config import cfg
import torch


class ProposalTargetLayer(nn.Module):
    def __init__(self):
        super(ProposalTargetLayer, self).__init__()
        self.rois_per_image = cfg.TRAIN.BATCH_SIZE
        self.fg_rois_per_image = int(
            np.round(cfg.TRAIN.FG_FRACTION * self.rois_per_image))

        self.fg_rois_per_image = int(
            np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    def forward(self, all_rois, gt_boxes, num_boxes):
        	"""
            Parameters
            ----------
            rpn_rois:  (1 x H x W x A, 5) [batch_index, x1, y1, x2, y2]
            gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] in
            """
        pass

    def _sample_rois(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        '''
        input:
            - inside_anchors: [number_of_anchor * 4]
            - batch_size: n
            - inside_anchor_indexs: [number_of_anchor * 1]
            - batch_boxes: list all ground truth boxes across all the image in batch
            - batch_boxes_index: batch_index of correspond with each boxes
                example: [0, 0, 0, 1, 1, 2, 2]
        '''

        # Caculate overlap bw all_rois and gt_boxes
