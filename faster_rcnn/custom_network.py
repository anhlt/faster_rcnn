from .faster_rcnn import FastRCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import FC


class CustomFasterRCNN(FastRCNN):
    """docstring for CustomFasterRCNN"""

    def __init__(self, origin_classes, multi_classes, debug=False):
        super(CustomFasterRCNN, self).__init__(origin_classes, debug)
        self.multi_classes = multi_classes
        self.n_multi_classes = len(multi_classes)
        self.debug = debug

        self.multiclass_fc = nn.DataParallel(FC(4096, self.n_multi_classes, relu=False))
