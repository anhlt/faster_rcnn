import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from .rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from .network import vgg16, Conv2d, np_to_variable, FC, tensor_to_variable
from roi_pooling.modules.roi_pool import RoIPool
from .fastrcnn.bbox_transform import bbox_transform_inv, clip_boxes
from .fastrcnn.nms_wrapper import nms
from .utils.blob import im_list_to_blob
import cv2
from network import smooth_l1_loss
from .rpn_msr.generate_anchors import generate_anchors


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [4, 8, 16, 32]

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
        return self.cross_entropy + 10 * self.loss_box

    def _computer_forward(self, im_data):
                # im_data = im_data.permute(0, 3, 1, 2)  # (N, 3, W, H)
        im_data = tensor_to_variable(im_data)
        features = self.features(im_data)  # (N, 512, W, H)
        rpn_conv1 = self.conv1(features)  # (N, 512, W, H)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)  # (N, A * 2, W, H)
        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        return features, rpn_bbox_pred, rpn_cls_score

    def forward(self,
                im_data,
                im_info, gt_boxes=None):
        features, rpn_bbox_pred, rpn_cls_score = self._computer_forward(im_data)

        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)  # (N, 2, -1, H)
        # (N, 2, -1, H) # Because Softmax take place over dimension 1
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales) * 3 * 2)  # (N , H , W , Ax2)
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred,
                                   im_info, cfg_key, self._feat_stride,
                                   self.anchor_scales)

        if self.training:
            assert gt_boxes is not None
            # list GT boxes
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes,
                                                im_info,
                                                self._feat_stride,
                                                self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(
                rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return features, rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(
            0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_label = rpn_data[0].view(-1)

        rpn_keep = Variable(
            rpn_label.data.ne(-1).nonzero().squeeze()).cuda()

        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, bbox_outside_weights = rpn_data[1:]
        rpn_loss_box = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, bbox_outside_weights, sigma=3.0)
        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred,
                              im_info, cfg_key, _feat_stride, anchor_scales)
        x = np_to_variable(x, is_cuda=True)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, im_info, _feat_stride, anchor_scales)

        rpn_labels = np_to_variable(
            rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = np_to_variable(
            rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = np_to_variable(
            rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = np_to_variable(
            rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def predict(self, im_data, im_info):
        self.eval()

        features, rpn_bbox_pred, rpn_cls_score = self._computer_forward(im_data)
        height, width = rpn_cls_score.shape[-2:]
        rpn_bbox_pred =  rpn_bbox_pred.data.cpu().numpy()
        _anchors = generate_anchors(scales=np.array(self.anchor_scales))
        _num_anchors = _anchors.shape[0]
        im_info = im_info[0]
        bbox_deltas = rpn_bbox_pred
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = _num_anchors
        K = shifts.shape[0]
        anchors = _anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))


        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        return proposals


class FasterRCNN(nn.Module):
    """docstring for FasterRCNN"""

    SCALES = (600, )
    MAX_SIZE = 1000

    def __init__(self, classes, debug=False):
        super(FasterRCNN, self).__init__()
        assert classes is not None
        self.classes = classes
        self.n_classes = len(classes)

        self.features = vgg16()
        # self.rpn = RPN()
        self.roi_pool = RoIPool(7, 7, 1.0 / 16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

        self.cross_entropy = None
        self.loss_box = None

        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box

    def forward(self, im_data, im_info, gt_boxes=None):
        # features, rois = self.rpn(
        #     im_data, im_info, gt_boxes)

        # features.size = (1 , W, H, 512)
        # rois
        rois = gt_boxes[gt_boxes[:, 4] == 0.]
        rois = np.roll(rois, 1)
        rois = np_to_variable(rois)
        gt_boxes = gt_boxes[gt_boxes[:, 4] != 0.]
        if self.training:
            roi_data = self.proposal_target_layer(
                rois, gt_boxes, self.n_classes)
            rois = roi_data[0]

        im_data = np_to_variable(im_data)
        im_data = im_data.permute(0, 3, 1, 2)
        features = self.features(im_data)

        # Roi pool
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)
        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(
                cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, bbox_pred, roi_data):
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt
        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            # print predict
            self.tp = torch.sum(predict[:fg_cnt].eq(
                label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt
            print 'fg_cnt', fg_cnt
            print 'bg_cnt', bg_cnt
            print 'tp', self.tp
            print 'cls_score.size()', cls_score.size()

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]

        # # bounding box regression L1 loss
        # bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        # bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
        # if True:
        #     print 'bbox_targets 2', bbox_targets.cpu().data.numpy()
        #     print 'bbox_targets 2', bbox_targets.cpu().data.numpy().shape

        loss_box = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        return cross_entropy, loss_box

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        rpn_rois = rpn_rois.data.cpu().numpy()
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_py(
                rpn_rois, gt_boxes, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = np_to_variable(rois, is_cuda=True)
        labels = np_to_variable(
            labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = np_to_variable(
            bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = np_to_variable(
            bbox_outside_weights, is_cuda=True)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(
                pred_boxes, scores, 0.1, inds=inds)
        self.classes = np.array(self.classes)
        return pred_boxes, scores, self.classes[inds], boxes

    def detect(self, image, thr=0.5, rois=None):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes, rois = \
            self.interpret_faster_rcnn(
                cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr, nms=True)
        return pred_boxes, scores, classes, rois

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        # im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)
