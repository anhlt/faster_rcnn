import torch.nn as nn
import numpy as np
import numpy.random as npr
from torch.autograd import Variable
from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps
from ..fastrcnn.bbox_transform import bbox_transform
from ..config import cfg
from torch import Tensor
import torch
from ..network import np_to_variable
import logging

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)


class AnchorTargerLayer(nn.Module):

    """Calculate target for RPN network

    Attributes
    ----------
    is_cuda : bool
        Using GPU or not
    """

    def __init__(self, feat_stride, anchor_scales, is_cuda=True):
        """Summary

        Parameters
        ----------
        feat_stride : class::`numpy.array`
            The Ratio between original image size and the convolutional feature (feature maps),
            Used to calculate anchors from a point in convolutional feature into the original image.

            Example: np.array([16. ,])
        anchor_scales : class::`numpy.array`
            Description
        is_cuda : bool, optional
            Description
        """
        super(AnchorTargerLayer, self).__init__()
        self._feat_stride = feat_stride
        self._anchor_scales = anchor_scales
        self._anchors = generate_anchors(scales=np.array(self._anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self.is_cuda = is_cuda

        # allow boxes to sit over the edge by a small amount
        self._allow_border = 0

    def forward(self, rpn_cls_score, gt_boxes, batch_boxes_index, im_info):
        """ Generate all anchors, then filter anchors that lays outside original images. Calculate target base on overlaps values and bbox regression.

        Parameters
        ----------
        rpn_cls_score: :class:`torch.Tensor`
            The probability of each anchor contains object center
        gt_boxes: :class:`numpy.array`
            List all ground truth boxes across all the images in batch
        batch_boxes_index: :class:`numpy.array`
            Batch index where image belong to.
        im_info: :class:`torch.Tensor([[im_height, im_width]])`
            Original Image size

        Returns
        -------
        (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`)
            Return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights



        """
        rpn_cls_score = rpn_cls_score.cpu().detach().numpy()
        # gt_boxes = gt_boxes.numpy()
        # im_info = im_info.numpy()
        # batch_boxes_index = batch_boxes_index.numpy()
        batch_boxes = gt_boxes[:, :4]

        feature_height, feature_width = rpn_cls_score.shape[2], rpn_cls_score.shape[3]
        batch_size = rpn_cls_score.shape[0]
        im_height, im_width = im_info[0][0], im_info[0][1]

        A = self._num_anchors

        # 1. Generate proposal from bbox deltas and shifted anchors
        all_anchors = self._create_anchors(feature_height, feature_width)
        total_anchors = all_anchors.shape[0]

        # only keep anchors inside the image
        inside_anchors, inside_anchor_indexes = self._filter_outside_anchors(
            all_anchors, im_height, im_width)

        # 2. Calculate overlap and assign corresponding label

        bbox_inside_weights = np.zeros(
            (batch_size, inside_anchor_indexes.shape[0], 4), dtype=np.float32)
        bbox_outside_weights = np.zeros(
            (batch_size, inside_anchor_indexes.shape[0], 4), dtype=np.float32)

        labels, bbox_targets = self.calculate_target(
            inside_anchors, batch_size, inside_anchor_indexes, batch_boxes, batch_boxes_index)

        # 3. calculate bbox_inside_weights, bbox_outside_weights
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = np.sum(labels == 1, axis=1)
        sum_bg = np.sum(labels == 0, axis=1)

        for i in range(batch_size):
            current_batch_sum_fg = sum_fg[i]
            current_batch_sum_bg = sum_bg[i]
            current_batch_fg_index = np.where(labels[i] == 1)[0]
            current_batch_bg_index = np.where(labels[i] == 0)[0]
            if current_batch_sum_fg > num_fg:
                disable_inds = npr.choice(
                    current_batch_fg_index, size=(current_batch_sum_fg - num_fg), replace=False)
                labels[i][disable_inds] = -1

                # subsample negative labels if we have too many

            num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels[i] == 1)
            if current_batch_sum_bg > num_bg:
                disable_inds = npr.choice(
                    current_batch_bg_index, size=(current_batch_sum_bg - num_bg), replace=False)

                labels[i][disable_inds] = -1

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = np.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples
            negative_weights = 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (
                cfg.TRAIN.RPN_POSITIVE_WEIGHT / (np.sum(labels == 1) + 1))
            negative_weights = (
                (1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / (np.sum(labels == 0) + 1))

        bbox_outside_weights[labels == 1] = np.array([positive_weights] * 4)
        bbox_outside_weights[labels == 0] = np.array([negative_weights] * 4)

        labels = self._unmap(labels, total_anchors,
                             inside_anchor_indexes, batch_size, fill=-1)
        bbox_targets = self._unmap(bbox_targets, total_anchors,
                                   inside_anchor_indexes, batch_size, fill=0)
        bbox_inside_weights = self._unmap(
            bbox_inside_weights, total_anchors, inside_anchor_indexes, batch_size, fill=0)
        bbox_outside_weights = self._unmap(
            bbox_outside_weights, total_anchors, inside_anchor_indexes, batch_size, fill=0)

        labels = labels.reshape((batch_size, feature_height, feature_width, A))
        labels = labels.transpose((0, 3, 1, 2))
        labels = labels.reshape(
            (batch_size, 1, A * feature_height, feature_width))

        bbox_targets = bbox_targets.reshape(
            (batch_size, feature_height, feature_width, A * 4)).transpose((0, 3, 1, 2))
        bbox_inside_weights = bbox_inside_weights.reshape(
            (batch_size, feature_height, feature_width, A * 4)).transpose((0, 3, 1, 2))

        bbox_outside_weights = bbox_outside_weights.reshape(
            (batch_size, feature_height, feature_width, A * 4)).transpose((0, 3, 1, 2))

        return np_to_variable(labels, self.is_cuda, dtype=torch.LongTensor), np_to_variable(bbox_targets, self.is_cuda), np_to_variable(bbox_inside_weights, self.is_cuda), np_to_variable(bbox_outside_weights, self.is_cuda)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass

    def _create_anchors(self, feature_height, feature_width):
        """Create all anchors given features height, width.

        Parameters
        ----------
        feature_height : int
            feature map height
        feature_width : int
            feature map width

        Returns
        -------
        :class:`numpy.array`
            Anchors
        """
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
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    def _filter_outside_anchors(self, all_anchors, im_height, im_width):
        """Remove outside anchors from generated anchors

        Parameters
        ----------
        all_anchors : :class:`numpy.array`
            All generated anchors
        im_height : int
            Origin image height.
        im_width : int
            Origin image width

        Returns
        -------
        tuple(inside_anchors, index_of_inside_anchors)
            Return all inside anchors and its indexes
        """
        inds_inside = np.where(
            (all_anchors[:, 0] >= - self._allow_border) &
            (all_anchors[:, 1] >= - self._allow_border) &
            (all_anchors[:, 2] < im_width + self._allow_border) &  # width
            (all_anchors[:, 3] < im_height + self._allow_border)  # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        return anchors, inds_inside

    def _unmap(self, data, count, inds, batch_size, fill=0):
        if len(data.shape) == 2:
            ret = np.empty((batch_size, count), dtype=np.float32)
            ret.fill(fill)
            ret[:, inds] = data
        else:
            ret = np.empty(
                (batch_size, count, data.shape[2]), dtype=np.float32)
            ret.fill(fill)
            ret[:, inds, :] = data
        return ret

    def calculate_target(self, inside_anchors, batch_size, inside_anchor_indexes, batch_boxes, batch_boxes_index):
        """Calculate bbox_targets and layer


        Notes
        -----
        |   Create empty label array.
        |   - `label`:

            >>> label.shape
            (A, batch_size)

        |   For each batch, there are `A` anchors and `G` batch boxes:
        |
        |   `current_batch_overlaps`: overlaps between anchors and boxes

            >>> current_batch_overlaps.shape
            (A, G)

        |    `argmax_overlaps` : List index of boxes, that have largest overlap w.r.t each Anchor.

            >>> argmax_overlaps.shape
            (A, 1)

        |    `max_overlaps` : List of largest overlap values between boxes w.r.t each Anchor.

            >>> max_overlaps.shape
            (A, 1)

        |    Set current batch `label` values:

            >>> labels[i, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            >>> labels[i, max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        Parameters
        ----------
        inside_anchors : :class:`numpy.array`
            List all anchors that lay inside the original images.
            Shape: [number_of_anchor * 4]
        batch_size : int
            Current batch size
        inside_anchor_indexes : :class:`numpy.array`
            Indexes of inside anchors.
            Shape: [number_of_anchor * 1]
        batch_boxes : :class:`numpy.array`
            list all ground truth boxes across all the images in batch
            example:

            >>> batch_boxes
            [[  48.57142857  465.71428571  537.14285714 1774.28571429]
             [ 220.         1065.71428571  382.85714286 1771.42857143]
             [ 326.76056338  315.49295775  394.36619718  580.28169014]
             [  76.05633803  290.14084507  242.25352113  788.73239437]
             [  11.26760563    8.45070423  585.91549296 1769.01408451]
             [ 178.125       221.875       287.5         975.        ]
             [ 321.875       334.375       434.375       984.375     ]]

        batch_boxes_index : list[Int]
            |  Batch index where image belong to.
            |  Example:
            |  There 3 images in current batch, and 6 boxes inside that 3 images. Look at the `batch_boxes_index` we know:
            |  First 3 boxes belong to first image.
            |  Next 2 boxes belong to second image.
            |  Last box belong to last image.

            >>> [0, 0, 0, 1, 1, 2]

        Returns
        -------
        (numpy.array((A, batch_size)), numpy.array((batch_size, A, 4)))
            Return caculated labels , and bbox_targers
        """
        labels = np.empty(
            (batch_size, inside_anchor_indexes.shape[0]), dtype=np.float32)
        bbox_targets = np.zeros(
            (batch_size, inside_anchor_indexes.shape[0], 4), dtype=np.float32)
        labels.fill(-1)

        overlaps = bbox_overlaps(inside_anchors.astype(
            np.float), batch_boxes.astype(np.float))

        for i in range(batch_size):
            current_batch_overlaps = overlaps[:, batch_boxes_index == i]
            current_batch_boxes = batch_boxes[[batch_boxes_index == i]]

            argmax_overlaps = current_batch_overlaps.argmax(axis=1)  # (A)
            max_overlaps = current_batch_overlaps[np.arange(
                inside_anchor_indexes.shape[0]), argmax_overlaps]
            gt_argmax_overlaps = current_batch_overlaps.argmax(axis=0)  # G
            gt_max_overlaps = current_batch_overlaps[gt_argmax_overlaps, np.arange(
                current_batch_overlaps.shape[1])]
            gt_argmax_overlaps = np.where(
                current_batch_overlaps == gt_max_overlaps)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[i, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # fg label: for each gt, anchor with highest overlap
            labels[i, gt_argmax_overlaps] = 1
            # fg label: above threshold IOU


            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[i, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            current_batch_bbox_targets = bbox_transform(inside_anchors.astype(
                np.float), current_batch_boxes[argmax_overlaps, :]).astype(np.float32, copy=False)
            bbox_targets[i, :] = current_batch_bbox_targets

        return labels, bbox_targets
