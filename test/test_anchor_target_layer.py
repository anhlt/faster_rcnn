from faster_rcnn.rpn_msr.anchor_target_layer_2 import AnchorTargerLayer
import unittest


class AnchorTargerLayerTest(unittest.TestCase):
    """docstring for AnchorTargerLayerTest"""

    def setUp(self):
        self.anchor_scales = [8, 16, 32]
        self.layer = AnchorTargerLayer(
            [16, ], anchor_scales=self.anchor_scales)

    def test_dump(self):
        assert True

    def test_init(self):
        assert self.layer._feat_stride == [16, ]

    def test_create_anchors(self):
        feature_height, feature_width = 37, 50
        anchors = self.layer._create_anchors(feature_height, feature_width)
        assert anchors.shape[0] == 37 * 50 * len(self.anchor_scales) * 3

    def test_overlap(self):
        origin_height, origin_width = 1666, 600
        feature_height, feature_width = 104, 37

        boxes = [[256.66666667, 16.66666667, 530., 1436.66666667],
                 [140., 93.33333333, 346.66666667, 1270.]]

        
