import numpy as np
import torch
from collections import defaultdict

def convert_data(blobs):
    blobs = [i for i in blobs if i is not None]
    current_batch_size = len(blobs)
    if not current_batch_size:
       return None
    max_height = np.max([blob['tensor'].shape[2] for blob in blobs])
    max_width = np.max([blob['tensor'].shape[3] for blob in blobs])
    batch_tensor = torch.Tensor(
        current_batch_size, 3, max_height, max_width).fill_(0.)
    total_boxes = 0
    batch_boxes = np.empty((0, 5))
    batch_boxes_index = np.empty((0,), dtype=np.int)
    img_name = []
    im_info = np.array([[batch_tensor.shape[2], batch_tensor.shape[3]]])
    for i, blob in enumerate(blobs):
        shape = blob['tensor'].shape
        batch_tensor[i, :, :shape[2], :shape[3]] = blob['tensor']
        total_boxes = blob['boxes'].shape[0]
        gt_classes = blob['gt_classes']
        gt_boxes = np.hstack([blob['boxes'], gt_classes[:, np.newaxis]])
        batch_boxes = np.vstack((batch_boxes, gt_boxes))
        a = np.zeros((total_boxes, ), dtype=np.int)
        a.fill(i)
        batch_boxes_index = np.concatenate((batch_boxes_index, a), axis=0)
        img_name.append(blob['im_name'])

    return batch_tensor, im_info, batch_boxes, batch_boxes_index, img_name


def convert_data_with_out_img(blobs):

    dd = defaultdict(list)
    for i, blob in enumerate(blobs):
        for key, val in blob.iteritems():  # .items() in Python 3.
            dd[key].append(val)

    return dict(dd)
