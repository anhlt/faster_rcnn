from __future__ import division
import torch.utils.data as data
import os
from torchvision import transforms
import logging
from PIL import Image
import numpy as np


__all__ = ['OpenImage']


class SingleOpenImage(data.Dataset):
    def __init__(self, root, class_name, general_transform=None,
                 transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.general_transform = general_transform

        self.base_dir = os.path.join(root, class_name)
        self.label_dir = os.path.join(root, class_name, 'Label')

        self.files = [f for f in os.listdir(self.base_dir)
                      if os.path.isfile(os.path.join(self.base_dir, f)) and
                      f.endswith(".jpg")]

        self.labels = [f for f in os.listdir(self.label_dir)
                       if os.path.isfile(os.path.join(self.base_dir, f)) and
                       f.endswith(".txt")]

        self.filename = [f.split('.')[0] for f in self.files]

        self.classes = [class_name]

        self.label_map_dict = {}

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

        for label in self.labels:
            with open(os.path.join(self.label_dir, label)) as f:
                for line in f:
                    print line

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.filename[index % len(self.filename)]

        image_path = os.path.join(self.base_dir, (filename + ".jpg"))
        annotation_path = os.path.join(self.label_dir, (filename + ".txt"))

        img = Image.open(image_path).convert('RGB')

        origin_size = img.size

        def annotation(annotation_file):
            with open(annotation_file, "r") as fp:
                for line in fp:
                    values = line.split()
                    yield (map(float, values[1:]), values[0])

        gt_boxes, gt_classes = zip(*annotation(annotation_path))
        gt_boxes = np.array(gt_boxes, dtype=np.int32)
        gt_classes = np.array([self.label_map_dict[class_name]
                               for class_name in list(gt_classes)],
                              dtype=np.int32)

        if self.general_transform:
            img, gt_boxes = self.general_transform(img, gt_boxes)

        blobs = {}
        blobs['gt_classes'] = gt_classes
        blobs['boxes'] = gt_boxes
        blobs['tensor'] = self.transform(img).unsqueeze(0)

        target_size = tuple(img.size)
        im_info = np.array(
            [[float(target_size[0]),
              float(target_size[1]),
              min(target_size) / min(origin_size)]])

        blobs['im_info'] = im_info
        blobs['im_name'] = os.path.basename(filename + ".jpg")

        return blobs


class OpenImage(data.Dataset):
    def __init__(self, root, imageset, *args, **kwargs):
        self.root = root
        self.base_dir = os.path.join(self.root, imageset)

        self.label_map_dict = {'__background__': 0}
        self.classes = ['__background__']
        self.sub_class_dict = {}

        self.sum_item = [0]

        self.sub_dirs = [name for name in os.listdir(self.base_dir)
                         if os.path.isdir(os.path.join(self.base_dir, name)
                                          ) and
                         not name.startswith(".")]

        for id, sub_dir in enumerate(self.sub_dirs, 1):
            self.label_map_dict[sub_dir] = id
            self.classes.append(sub_dir)

        for sub_class in self.classes[1:]:
            self.sub_class_dict[sub_class] = SingleOpenImage(
                self.base_dir, sub_class, *args, **kwargs
            )
            self.sub_class_dict[sub_class].label_map_dict = self.label_map_dict

            self.sum_item.append(
                self.sum_item[-1] + len(self.sub_class_dict[sub_class]))

    def __getitem__(self, index):

        def find(a, value, start, end):
            mid = int((end + start) / 2)
            if end == start + 1:
                return end, value - a[start] - 1
            if a[mid] < value:
                return find(a, value, mid, end)
            else:
                return find(a, value, start, mid)

        category, offset = find(self.sum_item, index + 1,
                                0, len(self.sum_item) - 1)
        # logger.debug(category, offset)
        return self.sub_class_dict[self.classes[category]][offset]

    def __len__(self):
        return self.sum_item[-1]
