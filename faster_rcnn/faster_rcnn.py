import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from .rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from .rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from .network import vgg16, Conv2d, np_to_variable


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

    def forward(self,
                im_data,
                im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        im_data = np_to_variable(im_data)
        im_data = im_data.permute(0, 3, 1, 2)  # (N, 512, W, H)
        features = self.features(im_data)  # (N, 512, W, H)
        rpn_conv1 = self.conv1(features)  # (N, 512, W, H)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)  # (N, A * 2, W, H)
        rpn_cls_score_reshape = self.reshape_layer(
            rpn_cls_score, 2)  # (N, 2, -1, H)
        # (N, 2, -1, H) # Because Softmax take place over dimension 1
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(
            self.anchor_scales) * 3 * 2)  # (N , H , W , Ax2)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred,
                                   im_info, cfg_key, self._feat_stride,
                                   self.anchor_scales)

        if self.training:
            assert gt_boxes is not None
            # list GT boxes
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes,
                                                gt_ishard,
                                                dontcare_areas,
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

        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[
            1:]

        rpn_bbox_targets = torch.mul(
            rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_loss_box = F.smooth_l1_loss(
            rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

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
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
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
            anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard,
                                   dontcare_areas, im_info, _feat_stride, anchor_scales)

        rpn_labels = np_to_variable(
            rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = np_to_variable(
            rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = np_to_variable(
            rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = np_to_variable(
            rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
