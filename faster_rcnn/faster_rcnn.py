import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rpn_msr.proposal_layer import ProposalLayer
from .rpn_msr.anchor_target_layer import AnchorTargerLayer
from rpn_msr.proposal_target_layer import ProposalTargetLayer
from .network import vgg16, Conv2d, np_to_tensor, FC, smooth_l1_loss
from roi_pooling.modules.roi_pool import RoIPool
from .fastrcnn.bbox_transform import bbox_transform_inv, clip_boxes
from .fastrcnn.nms_wrapper import nms
from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):

    """Generate region proposals, shares computation with the object detection network.

    Attributes
    ----------
    anchor_scales : list
        The scale of each anchor on particular point on feature maps.
    anchor_target_layer : :class:`faster_rcnn.rpn_msr.anchor_target_layer.AnchorTargerLayer`
        Calculate network target base on anchors and ground truth boxes.
    bbox_conv : :class:`torch.nn.module`
        Proposals coordinate refine predictor
    conv1 : :class:`torch.nn.module`
        Probability that anchors contains object predictor
    cross_entropy : int
        Cross entropy loss.
    features : :class:`torch.nn.module`
        Backbone network, that share computation with object detection network
    loss_box : int
        Box coordinate refine loss.
    proposal_layer : :class:`faster_rcnn.rpn_msr.proposal_layer.ProposalLayer`
        Create proposals base on generated anchors and bbox refine values.
    score_conv : TYPE
        Description
    """

    _feat_stride = [16, ]
    anchor_scales = [4, 8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()
        self.features = vgg16()
        self.features = nn.DataParallel(self.features)
        self.conv1 = nn.DataParallel(Conv2d(512, 512, 3, same_padding=True))
        self.score_conv = nn.DataParallel(Conv2d(
            512, len(self.anchor_scales) * 3 * 2, 1, relu=False))
        self.bbox_conv = nn.DataParallel(Conv2d(
            512, len(self.anchor_scales) * 3 * 4, 1, relu=False))

        self.anchor_target_layer = AnchorTargerLayer(
            self._feat_stride, self.anchor_scales)
        self.proposal_layer = ProposalLayer(
            self._feat_stride, self.anchor_scales)

    def _computer_forward(self, im_data):
        """Calculate forward

        Parameters
        ----------
        im_data : :class:`torch.tensor`
            image as tensor

        Returns
        -------
        (:class:`torch.tensor`, :class:`torch.tensor`, :class:`torch.tensor`)
            Return feature map, proposal boxes refine values w.r.t to each anchors, probability that anchors is foreground
        """
        features = self.features(im_data)  # (N, 512, W, H)
        rpn_conv1 = self.conv1(features)  # (N, 512, W, H)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)  # (N, A * 2, W, H)
        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        return features, rpn_bbox_pred, rpn_cls_score

    def forward(self,
                im_data,
                im_info, gt_boxes=None, gt_boxes_index=[]):
        """Forward

        Parameters
        ----------
        im_data : TYPE
            Description
        im_info : TYPE
            Description
        gt_boxes : None, optional
            Description
        gt_boxes_index : list, optional
            Description

        Returns
        -------
        tuple(features, rois)
            Return the features map and list of rois.
        """

        im_data = im_data.to(torch.device('cuda'))
        features, rpn_bbox_pred, rpn_cls_score = self._computer_forward(
            im_data)
        batch_size = features.shape[0]

        # rpn_cls_score : batch ,(num_anchors * 2) , h ,w = 1 , (4 * 3 * 2) , h , w

        rpn_cls_score_reshape = rpn_cls_score.view(
            batch_size, 2, -1, rpn_cls_score.shape[-1])  # batch , 2 , (num_anchors*h) , w
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob_reshape = rpn_cls_prob.view_as(
            rpn_cls_score)  # batch , h , w , (num_anchors * 2)

        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred,
                                   im_info, cfg_key)

        if self.training:
            assert gt_boxes is not None
            # list GT boxes
            target = self.anchor_target_layer(
                rpn_cls_score, gt_boxes, gt_boxes_index, im_info)
            # self.cross_entropy, self.loss_box = self.build_loss(
            #     rpn_cls_score_reshape, rpn_bbox_pred, target)
        else:
            target = None

        return features, rois, rpn_cls_prob_reshape, rpn_bbox_pred, target

    @staticmethod
    def build_loss(rpn_cls_score_reshape, rpn_bbox_pred, target):
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # batch * h * w * a , 2
        rpn_label = target[0].permute(0, 2, 3, 1).contiguous().view(-1)

        rpn_keep = torch.tensor(
            rpn_label.data.ne(-1).nonzero().squeeze()).cuda()

        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, bbox_outside_weights = target[1:]
        rpn_loss_box = smooth_l1_loss(
            rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])
        return rpn_cross_entropy, rpn_loss_box

    def predict_rois(self, im_data, im_info):
        self.eval()
        _, rois = self(im_data, im_info)
        return rois


class FastRCNN(nn.Module):
    """docstring for FasterRCNN

    Attributes
    ----------
    bbox_fc : TYPE
        Description
    classes : TYPE
        Description
    cross_entropy : TYPE
        Description
    debug : TYPE
        Description
    fc6 : TYPE
        Description
    fc7 : TYPE
        Description
    loss_box : TYPE
        Description
    MAX_SIZE : int
        Description
    n_classes : TYPE
        Description
    proposal_target_layer : TYPE
        Description
    roi_pool : TYPE
        Description
    rpn : TYPE
        Description
    SCALES : tuple
        Description
    score_fc : TYPE
        Description
    """

    SCALES = (600, )
    MAX_SIZE = 1000

    def __init__(self, classes, debug=False):
        super(FastRCNN, self).__init__()
        assert classes is not None
        self.classes = classes
        self.n_classes = len(classes)

        # self.features = vgg16()
        self.rpn = RPN()
        self.proposal_target_layer = ProposalTargetLayer(self.n_classes)
        self.roi_pool = RoIPool(7, 7, 1.0 / 16)
        self.fc6 = nn.DataParallel(FC(512 * 7 * 7, 4096))
        self.fc7 = nn.DataParallel(FC(4096, 4096))
        self.score_fc = nn.DataParallel(FC(4096, self.n_classes, relu=False))
        self.bbox_fc = nn.DataParallel(
            FC(4096, self.n_classes * 4, relu=False))

        self.debug = debug

    def forward(self, im_data, im_info, gt_boxes=None, gt_boxes_index=[]):
        """Summary

        Parameters
        ----------
        im_data : TYPE
            Description
        im_info : TYPE
            Description
        gt_boxes : None, optional
            Description
        gt_boxes_index : list, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        features, rois, rpn_cls_prob_reshape, rpn_bbox_pred, rpn_target = self.rpn(
            im_data, im_info, gt_boxes, gt_boxes_index)

        if self.training:
            target = self.proposal_target_layer(
                rois, gt_boxes, gt_boxes_index)
            rois = target[0]
        else:
            target = None
            rois = rois.reshape(-1, 5).type(torch.FloatTensor).to(torch.device("cuda"))

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

        return cls_prob, bbox_pred, rois, cls_score, target, rpn_cls_prob_reshape, rpn_bbox_pred, rpn_target

    @staticmethod
    def build_loss(cls_score, bbox_pred, target):
        label = target[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt.item()) / bg_cnt.item()
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)
        bbox_targets, bbox_inside_weights, bbox_outside_weights = target[2:]

        loss_box = smooth_l1_loss(
            bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, dim=[1])
        return cross_entropy, loss_box

    def interpret(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
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
        boxes = rois.data.cpu().numpy()[keep, 1:5]
        if len(boxes) == 0:
            return np.array([]), np.array([]),np.array([]),np.array([])
        pred_boxes = bbox_transform_inv(
            boxes[np.newaxis, :], box_deltas[np.newaxis, :])
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)
        pred_boxes = pred_boxes[0]
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(
                pred_boxes, scores, 0.2, inds=inds)
        self.classes = np.array(self.classes)
        return pred_boxes, scores, self.classes[inds], boxes

    def detect(self, image, thr=0.5):
        self.eval()
        im_data, im_info = self.get_image_blob(image)
        cls_prob, bbox_pred, rois, _, _, _, _, _ = self(im_data, im_info[:, :2])
        cls_prob = cls_prob.squeeze()
        bbox_pred = bbox_pred.squeeze()
        pred_boxes, scores, classes, rois = \
            self.interpret(
                cls_prob, bbox_pred, rois, im_info, im_info[0][:2], min_score=thr, nms=True)
        return pred_boxes, scores, classes, rois, im_data

    def detect_blob(self, im_data, im_info, thr=0.5):
        self.eval()
        cls_prob, bbox_pred, rois, _, _, _, _, _ = self(im_data, im_info[:, :2])
        cls_prob = cls_prob.squeeze()
        bbox_pred = bbox_pred.squeeze()
        pred_boxes, scores, classes, rois = \
            self.interpret(
                cls_prob, bbox_pred, rois, im_info, im_info[0][:2], min_score=thr, nms=True)
        return pred_boxes, scores, classes, rois, im_data

    def get_image_blob(self, im):
        """Converts an image into a network input.

        Parameters
        ----------
        im : ndarray
            a color image in BGR order

        Returns
        -------
        blob : ndarray
            a data blob holding an image pyramid
        im_scale_factors : list
            list of image scales (relative to im) used
            in the image pyramid
        """
        transform = transforms.Compose([
            transforms.Resize(600),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225])])

        img = Image.open(im).convert('RGB')
        origin_size = img.size
        img = transform(img)
        img = img.unsqueeze(0)
        target_size = tuple(img.size())
        im_info = np.array(
            [[float(target_size[2]), float(target_size[3]), 600. / min(origin_size)]])

        return img, im_info
