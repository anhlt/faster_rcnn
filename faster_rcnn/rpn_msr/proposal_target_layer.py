import torch.nn as nn
import numpy as np
import numpy.random as npr

from ..fastrcnn.bbox_transform import bbox_transform
from ..utils.cython_bbox import bbox_overlaps
from ..config import cfg
from ..network import np_to_tensor
import torch
import logging

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)


class ProposalTargetLayer(nn.Module):
    def __init__(self, num_classes, is_cuda=True):
        super(ProposalTargetLayer, self).__init__()
        self.rois_per_image = cfg.TRAIN.BATCH_SIZE

        self.fg_rois_per_image = int(
            np.round(cfg.TRAIN.FG_FRACTION * self.rois_per_image))
        self.num_classes = num_classes
        self.is_cuda = is_cuda

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

        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois(
            all_rois, gt_boxes, gt_boxes_index, self.fg_rois_per_image, self.rois_per_image, self.num_classes)

        rois = rois.reshape(-1, 5)
        labels = labels.reshape(-1, 1)
        bbox_targets = bbox_targets.reshape(-1, self.num_classes * 4)
        bbox_inside_weights = bbox_inside_weights.reshape(
            -1, self.num_classes * 4)

        bbox_outside_weights = np.array(
            bbox_inside_weights > 0).astype(np.float32)

        return rois, labels, bbox_targets, bbox_inside_weights, np_to_tensor(bbox_outside_weights)

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

        batch_size = all_rois.shape[0]

        rois = np.zeros((batch_size, self.rois_per_image, 5))
        bbox_targets = np.zeros(
            (batch_size, self.rois_per_image, self.num_classes * 4))
        bbox_inside_weights = np.zeros(
            (batch_size, self.rois_per_image, self.num_classes * 4))
        labels = np.zeros((batch_size, self.rois_per_image))

        for i in range(batch_size):
            current_rois = all_rois[i]
            current_gt_boxes = gt_boxes[[gt_boxes_index == i]]

            # overlaps : R x G
            overlaps = bbox_overlaps(current_rois[:, 1:], current_gt_boxes)
            gt_assignment = overlaps.argmax(axis=1)  # R
            max_overlaps = overlaps.max(axis=1)  # R
            current_labels = current_gt_boxes[gt_assignment, 4]
            fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
            fg_rois_per_this_image = min(self.fg_rois_per_image, fg_inds.size)
            if fg_inds.size > 0:
                fg_inds = npr.choice(
                    fg_inds, size=fg_rois_per_this_image, replace=False)

            bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (
                max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

            bg_rois_per_this_image = self.rois_per_image - fg_rois_per_this_image
            bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

            if bg_inds.size > 0:
                bg_inds = npr.choice(
                    bg_inds, size=bg_rois_per_this_image, replace=False)
            keep_inds = np.append(fg_inds, bg_inds)

            # Select sampled values from various arrays:
            current_labels = current_labels[keep_inds]
            current_labels[fg_rois_per_this_image:] = 0
            current_rois = current_rois[keep_inds]

            # logger.debug(gt_boxes.shape)
            # logger.debug(current_gt_boxes.shape)

            current_bbox_target_data = self._compute_targets(
                current_rois[:, 1:5], current_gt_boxes[gt_assignment[keep_inds], :4], current_labels)

            current_bbox_targets, current_bbox_inside_weights = self._get_bbox_regression_labels(
                current_bbox_target_data, num_classes)

            rois[i] = current_rois
            bbox_targets[i] = current_bbox_targets
            bbox_inside_weights[i] = current_bbox_inside_weights
            labels[i] = current_labels

        return np_to_tensor(labels, is_cuda=self.is_cuda, dtype=torch.LongTensor), np_to_tensor(rois, is_cuda=self.is_cuda), np_to_tensor(bbox_targets, is_cuda=self.is_cuda), np_to_tensor(bbox_inside_weights, is_cuda=self.is_cuda)

    def _compute_targets(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        ) / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
        return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    def _get_bbox_regression_labels(self, bbox_target_data, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """

        clss = bbox_target_data[:, 0]
        bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]
        for ind in inds:
            cls = int(clss[ind])
            start = 4 * cls
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        return bbox_targets, bbox_inside_weights
