import torch
from torch.autograd import Function
from .._ext import roi_pooling
from torch import Tensor


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_width, pooled_height, spatial_scale):
        pooled_width_int = int(pooled_width.item())
        pooled_height_int = int(pooled_height.item())
        spatial_scale_float = float(spatial_scale.item())

        batch_size, num_channels, data_height, data_width = features.shape
        num_rois = rois.shape[0]
        output = torch.zeros(num_rois, num_channels,
                             pooled_height_int, pooled_width_int)
        argmax = torch.IntTensor(
            num_rois, num_channels, pooled_height_int, pooled_width_int).zero_()

        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(pooled_height_int, pooled_width_int, spatial_scale_float,
                                            _features, rois, output)
            # output = output.cuda()
        else:
            output = output.cuda()
            argmax = argmax.cuda()
            roi_pooling.roi_pooling_forward_cuda(pooled_height_int, pooled_width_int, spatial_scale_float,
                                                 features, rois, output, argmax)

        ctx.save_for_backward(features, output, argmax, rois, pooled_width, pooled_height, spatial_scale)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, output, argmax, rois, pooled_width, pooled_height, spatial_scale = ctx.saved_tensors
        pooled_width = int(pooled_width.item())
        pooled_height = int(pooled_height.item())
        spatial_scale = float(spatial_scale.item())
        feature_size = features.shape

        assert(feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size

        grad_input = torch.zeros(
            batch_size, num_channels, data_height, data_width).cuda()
        roi_pooling.roi_pooling_backward_cuda(pooled_height, pooled_width, spatial_scale,
                                              grad_output, rois, grad_input, argmax)

        # print grad_input

        return grad_input, None, None, None, None, None, None
