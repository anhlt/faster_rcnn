from torchvision.datasets import CocoDetection
from PIL import Image
import os
import numpy as np


class CocoData(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(CocoData, self).__init__(
            root, annFile, transform, target_transform)
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
        width = image_info['width']
        height = image_info['height']
        coco_cat_id_to_class_index = dict([(self._class_to_coco_cat_id[
                                          class_name], self._class_to_index[class_name]) for class_name in self._classes[1:]])

        def bboxs(targets):
            for target in targets:
                x1 = np.max((0, target['bbox'][0]))
                y1 = np.max((0, target['bbox'][1]))
                x2 = np.min(
                    (width - 1, x1 + np.max((0, target['bbox'][2] - 1))))
                y2 = np.min(
                    (height - 1, y1 + np.max((0, target['bbox'][3] - 1))))
                class_index = coco_cat_id_to_class_index[target['category_id']]
                if target['area'] > 0 and x2 >= x1 and y2 >= y1:
                    yield [x1, y1, x2, y2, class_index]

        objects = [box for box in bboxs(annotation)]
        num_objs = len(objects)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for index, obj in annotation:
            seg_areas[index] = annotation['area']
            if obj['iscrowd']:
                overlaps[index, :] = -1.0
            else:
                class_index = coco_cat_id_to_class_index[obj['category_id']]
                overlaps[index, class_index] = 1.0

        

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, anns


def prepare_roidb(imdb):
    # cache file

    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = 'load path'
        boxes = roidb[i]['boxes']
        labels = roidb[i]['gt_classes']
        info_boxes = np.zeros((0, 18), dtype=np.float32)
