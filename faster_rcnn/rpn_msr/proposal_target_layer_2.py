import torch.nn as nn
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from ..fastrcnn.bbox_transform import bbox_transform_inv, clip_boxes
from ..fastrcnn.nms_wrapper import nms
from ..utils.cython_bbox import bbox_overlaps
from ..config import cfg
import torch


class ProposalTargetLayer(nn.Module):
    def __init__(self, num_classes):
        super(ProposalTargetLayer, self).__init__()
        self.rois_per_image = cfg.TRAIN.BATCH_SIZE

        self.fg_rois_per_image = int(
            np.round(cfg.TRAIN.FG_FRACTION * self.rois_per_image))
        self.num_classes = num_classes

    def forward(self, all_rois, gt_boxes, gt_boxes_index):
        """size
        Parameters
        ----------
        rpn_rois:  (1 x H x W x A, 5) [batch_index, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class]
        gt_boxes_index: batch_index of correspond with each boxes
            example: [0, 0, 0, 1, 1, 2, 2]
        """
        all_rois = all_rois.cpu().detach().numpy()

        return self._sample_rois(
            all_rois, gt_boxes, gt_boxes_index, self.fg_rois_per_image, self.rois_per_image, self.num_classes)

    def _sample_rois(self, all_rois, gt_boxes, gt_boxes_index, fg_rois_per_image, rois_per_image, num_classes):
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
        print(all_rois.shape)
        print(gt_boxes.shape)

        batch_size = all_rois.shape[0]

        for i in range(batch_size):
            print "Batch: %d" % i
            current_rois = all_rois[i]
            print "gt_boxes", gt_boxes.shape
            current_gt_boxes = gt_boxes[[gt_boxes_index == i]]

            # overlaps : R x G
            overlaps = bbox_overlaps(current_rois, current_gt_boxes)
            print "overlaps.shape", overlaps.shape
            gt_assignment = overlaps.argmax(axis=1)  # R
            max_overlaps = overlaps.max(axis=1)  # R
            print current_gt_boxes.shape
            labels = current_gt_boxes[gt_assignment, 4]
            fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
            print max_overlaps[fg_inds]
            fg_rois_per_this_image = min(self.fg_rois_per_image, fg_inds.size)
            if fg_inds.size > 0:
                fg_inds = npr.choice(
                    fg_inds, size=fg_rois_per_this_image, replace=False)

            print fg_inds

            bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (
                max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            print max_overlaps

            bg_rois_per_this_image = self.rois_per_image - fg_rois_per_this_image
            bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

            if bg_inds.size > 0:
                bg_inds = npr.choice(
                    bg_inds, size=bg_rois_per_this_image, replace=False)
            print bg_inds
            keep_inds = np.append(fg_inds, bg_inds)
            print 'keep_inds', keep_inds

            # Select sampled values from various arrays:
            labels = labels[keep_inds]
            labels[fg_rois_per_this_image:] = 0
            rois = current_rois[keep_inds]
            print rois.shape
            print "---"


