from torchvision.datasets import CocoDetection
import os
import numpy as np
from .blob import im_list_to_blob, prep_im_for_blob
from scipy.misc import imread
from ..config import cfg
import scipy.io as io
import scipy
from .ds_utils import unique_boxes, filter_small_boxes, validate_boxes
import logging
from ..utils.cython_bbox import bbox_overlaps
from torchvision import transforms
from torchvision.transforms import Resize
from PIL import Image

logger = logging.getLogger(__name__)


def _get_image_blob(im):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    processed_ims = []
    im_scales = []

    target_size = cfg.TRAIN.SCALES[0]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]])

    return blob, im_info


class CocoData(CocoDetection):
    def __init__(self,
                 root,
                 annFile,
                 pre_proposal_folder=None,
                 transform=None,
                 target_transform=None):
        super(CocoData, self).__init__(
            root, annFile, transform, target_transform)

        self.pre_proposal_folder = pre_proposal_folder
        self._data_name = 'train2014'
        self.config = {'top_k': 2000,
                       'use_salt': True,
                       'cleanup': True,
                       'crowd_thresh': 0.7,
                       'min_size': 2}
        self.root = root
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple(['__background__'] +
                              [cat['name'] for cat in cats])
        self._class_to_index = dict(
            zip(self.classes, xrange(len(self.classes))))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                              self.coco.getCatIds()))
        self._coco_cat_id_to_class_index = dict([(self._class_to_coco_cat_id[
            class_name], self._class_to_index[class_name]) for class_name in self._classes[1:]])

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(600),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

    def _get_box_file(self, index):
        # first 14 chars / first 22 chars / all chars + .mat
        # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        file_name = ('COCO_' + self._data_name +
                     '_' + str(index).zfill(12) + '.mat')
        return os.path.join(file_name[:14], file_name[:22], file_name)

    @property
    def classes(self):
        return self._classes

    def _load_image_set_index(self):
        return self.coco.getImgIds()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_id = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        annotation = coco.loadAnns(ann_id)
        image_info = coco.loadImgs(img_id)[0]
        im_file_path = os.path.join(self.root, image_info['file_name'])
        width = image_info['width']
        height = image_info['height']
        # need to fixed

        # im_blob, im_info = _get_image_blob(imread(im_file_path))
        img = Image.open(im_file_path).convert('RGB')
        origin_size = img.size
        img = self.transform(img)
        img = img.unsqueeze(0)
        target_size = tuple(img.size())
        im_info = np.array(
            [[float(target_size[2]), float(target_size[3]), 600. / min(origin_size)]])

        blobs = {}
        blobs['tensor'] = img
        blobs['im_name'] = os.path.basename(im_file_path)
        blobs['image_info'] = image_info

        # The standard in computer vision is to specify the top left corner and the bottom right corner.
        # The coordinates are parsed by <your_dataset.py> (for example coco.py)
        # in the function

        def bboxs(targets):
            for target in targets:
                overlap = np.array([0] * len(self.classes))
                x1 = np.max((0, target['bbox'][0]))
                y1 = np.max((0, target['bbox'][1]))
                x2 = np.min(
                    (width - 1, x1 + np.max((0, target['bbox'][2] - 1))))
                y2 = np.min(
                    (height - 1, y1 + np.max((0, target['bbox'][3] - 1))))
                class_index = self._coco_cat_id_to_class_index[
                    target['category_id']]
                if target['iscrowd']:
                    overlap[:] = -1.0
                else:
                    overlap[class_index] = 1.0
                if target['area'] > 0 and x2 >= x1 and y2 >= y1:
                    yield [x1, y1, x2, y2], class_index, target['area'], overlap
        try:
            gt_boxes, gt_classes, gt_seg_areas, gt_overlaps = zip(
                *[box for box in bboxs(annotation)])
        except ValueError:
            return None

        gt_boxes = np.array(gt_boxes, dtype=np.uint16)
        gt_classes = np.array(gt_classes, dtype=np.int32)
        gt_overlaps = np.array(gt_overlaps, dtype=np.float32)
        gt_seg_areas = np.array(gt_seg_areas, dtype=np.float32)

        # load pre-computed proposal boxes
        if self.pre_proposal_folder:
            box_file = os.path.join(
                self.pre_proposal_folder, 'mat', self._get_box_file(img_id))
            raw_data = io.loadmat(box_file)['boxes']
            boxes = np.maximum(raw_data - 1, 0).astype(np.uint16)
            boxes = boxes[:, (1, 0, 3, 2)]
            keep = unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            boxes = boxes[:self.config['top_k'], :]
            validate_boxes(boxes, width=width, height=height)

            box_overlaps = np.zeros((boxes.shape[0], len(self.classes)))
            box_classes = np.zeros((boxes.shape[0],), dtype=np.int32)
            box_seg_areas = np.zeros((boxes.shape[0],), dtype=np.float32)
            overlaps_with_gt = bbox_overlaps(boxes.astype(np.float),
                                             gt_boxes.astype(np.float))
            argmaxes = overlaps_with_gt.argmax(axis=1)
            maxes = overlaps_with_gt.max(axis=1)
            i = np.where(maxes > 0)[0]
            box_overlaps[i, gt_classes[argmaxes[i]]] = maxes[i]

            gt_boxes = np.vstack([gt_boxes, boxes])
            gt_classes = np.hstack([gt_classes, box_classes])
            gt_overlaps = np.vstack([gt_overlaps, box_overlaps])
            gt_seg_areas = np.hstack([gt_seg_areas, box_seg_areas])

        blobs['gt_ishard'] = np.zeros(len(gt_boxes))
        blobs['gt_classes'] = gt_classes
        blobs['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
        blobs['boxes'] = gt_boxes * im_info[0][2]
        blobs['flipped'] = False
        blobs['dontcare_areas'] = np.zeros([0, 4], dtype=np.float)
        blobs['im_info'] = np.array(im_info, dtype=np.float32)
        return blobs
