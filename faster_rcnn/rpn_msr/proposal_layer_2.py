import torch.nn as nn
import numpy as np
from .generate_anchors import generate_anchors
from ..fastrcnn.bbox_transform import bbox_transform_inv, clip_boxes
from ..fastrcnn.nms_wrapper import nms
from ..config import cfg
import torch
import logging

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)


class ProposalLayer(nn.Module):

    def __init__(self, _feat_stride=[16, ], anchor_scales=[8, 16, 32]):
        super(ProposalLayer, self).__init__()
        self._feat_stride = _feat_stride
        self._anchor_scales = anchor_scales
        self._anchors = generate_anchors(scales=np.array(self._anchor_scales))
        self._num_anchors = self._anchors.shape[0]

    def forward(self, scores, bbox_deltas, im_info, cfg_key):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        scores = scores.cpu().detach().numpy()
        bbox_deltas = bbox_deltas.cpu().detach().numpy()
        scores = scores[:, self._num_anchors:, :, :]
        im_info = im_info[0]

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        # min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = scores.shape[0]

        feature_height, feature_width = scores.shape[2], scores.shape[3]
        anchors = self._create_anchors(feature_height, feature_width)
        anchors = np.tile(anchors, (batch_size, 1, 1))

        bbox_deltas = bbox_deltas.transpose(
            (0, 2, 3, 1)).reshape((batch_size, -1, 4))
        scores = scores.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 1))
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, im_info)
        scores_keep = scores
        proposals_keep = proposals
        order = scores_keep.reshape((batch_size, -1)).argsort(axis=1)[:, ::-1]

        output = np.zeros((batch_size, post_nms_topN, 5))
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            order_single = order[i]
            order_single = order_single[:pre_nms_topN].ravel()
            proposals_single = proposals_single[order_single]
            scores_single = scores_single[order_single]
            keep = nms(np.hstack((proposals_single, scores_single)), nms_thresh)
            keep = keep[:post_nms_topN]
            logger.debug(order_single[keep[:10]])

            proposals_single = proposals_single[keep]
            scores_single = scores_single[keep]
            num_proposal = proposals_single.shape[0]
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single

        output = torch.from_numpy(output)
        return output

    def backward(self):
        pass

    def _create_anchors(self, feature_height, feature_width):
        shift_x = np.arange(0, feature_width) * self._feat_stride
        shift_y = np.arange(0, feature_height) * self._feat_stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # generate shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]

        # move to specific gpu.
        # self._anchors = self._anchors.type_as(gt_boxes)

        # add bbox deltas to shifted anchors to get proposal
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((1, K * A, 4))
        return all_anchors

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep
