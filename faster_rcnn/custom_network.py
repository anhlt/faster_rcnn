from .faster_rcnn import FastRCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import FC


class CustomFasterRCNN(nn.Module):
    """docstring for CustomFasterRCNN"""

    def __init__(self, origin_classes, multi_classes, debug=False):
        super(CustomFasterRCNN, self).__init__()
        self.multi_classes = multi_classes
        self.faster_rcnn = FastRCNN(origin_classes, debug)
        self.n_multi_classes = len(multi_classes)
        self.debug = debug
        self.multiclass_fc = nn.DataParallel(FC(4096, self.n_multi_classes, relu=False))

    def forward(self, im_data, im_info, gt_boxes=None, gt_boxes_index=[]):
        cls_prob, bbox_pred, rois, cls_score, target, rpn_cls_prob_reshape, rpn_bbox_pred, rpn_target, feature = super(CustomFasterRCNN, self).forward(im_data, im_data, im_info, gt_boxes, gt_boxes_index)

        multi_classes_score = self.multiclass_fc(feature)
        multi_classes_prob = F.softmax(multi_classes_score, dim=1)
        return bbox_pred, cls_prob, multi_classes_prob

    @staticmethod
    def build_loss(bbox_pred, cls_prob, multi_classes_prob, target):

        pass
