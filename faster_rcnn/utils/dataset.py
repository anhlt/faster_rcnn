from torchvision.datasets import CocoDetection
from PIL import Image
import os
import copy
import numpy as np


class CocoData(CocoDetection):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        def bboxs(target):
            for tar in target:
                tmp = copy.copy(tar['bbox'])
                tmp.append(tar['category_id'])
                yield tmp

        target = np.vstack((bbox for bbox in bboxs(anns)))

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
