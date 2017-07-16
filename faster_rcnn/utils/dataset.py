from torchvision.datasets import CocoDetection
import os
import numpy as np
from .blob import im_list_to_blob, prep_im_for_blob
import cv2
from ..config import cfg


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
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoData, self).__init__(
            root, annFile, transform, target_transform)

        self.root = root
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple(['__background__'] +
                              [cat['name'] for cat in cats])
        self._class_to_index = dict(
            zip(self.classes, xrange(len(self.classes))))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                              self.coco.getCatIds()))

    @property
    def classes(self):
        return self._classes

    def _load_image_set_index(self):
        return self.coco.getImgIds()

    # def _get_widths(self):
    #     anns = self._COCO.loadImgs(self._image_index)
    #     widths = [ann['width'] for ann in anns]
    #     return widths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_id = coco.getAnnIds(imgIds=img_id)
        annotation = coco.loadAnns(ann_id)
        image_info = coco.loadImgs(img_id)[0]
        im_file_path = os.path.join(self.root, image_info['file_name'])
        # need to fixed
        im_blob, im_info = _get_image_blob(cv2.imread(im_file_path))
        blobs = {'data': im_blob}
        blobs['im_name'] = os.path.basename(im_file_path)

        width = image_info['width']
        height = image_info['height']
        coco_cat_id_to_class_index = dict([(self._class_to_coco_cat_id[
                                          class_name], self._class_to_index[class_name]) for class_name in self._classes[1:]])

        # The standard in computer vision is to specify the top left corner and the bottom right corner.
        # The coordinates are parsed by <your_dataset.py> (for example coco.py) in the function
        def bboxs(targets):
            for target in targets:
                x1 = np.max((0, target['bbox'][0]))
                y1 = np.max((0, target['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, target['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, target['bbox'][3] - 1))))
                class_index = coco_cat_id_to_class_index[target['category_id']]
                if target['area'] > 0 and x2 >= x1 and y2 >= y1:
                    yield [x1, y1, x2, y2, class_index]

        objects = [box for box in bboxs(annotation)]
        blobs['gt_boxes'] = np.array(objects)
        num_objs = len(objects)
        overlaps = np.zeros((num_objs, len(self.classes)), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for index, obj in enumerate(annotation):
            seg_areas[index] = obj['area']
            if obj['iscrowd']:
                overlaps[index, :] = -1.0
            else:
                class_index = coco_cat_id_to_class_index[obj['category_id']]
                overlaps[index, class_index] = 1.0

        blobs['gt_ishard'] = np.zeros(len(objects))
        blobs['dontcare_areas'] = np.zeros([0, 4], dtype=np.float)
        blobs['im_info'] = np.array(im_info, dtype=np.float32)

        return blobs
