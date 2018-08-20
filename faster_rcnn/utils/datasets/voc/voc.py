import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
from torchvision import transforms
import numpy as np
import logging
try:
    from .string_int_label_map_pb2 import StringIntLabelMap
except Exception as e:
    from string_int_label_map_pb2 import StringIntLabelMap

from google.protobuf import text_format

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TransformVOCDetectionAnnotation(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []

        for obj in target.iter('object'):
            name = obj.find('name').text
            bb = obj.find('bndbox')
            bndbox = [bb.find('xmin').text, bb.find('ymin').text,
                      bb.find('xmax').text, bb.find('ymax').text]

            res += [bndbox + [name]]

        return res


class VOCSegmentation(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id)  # .convert('RGB')

        img = Image.open(self._imgpath % img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')
        self._label_map_path = os.path.join(self.root, dataset_name, 'pascal_label_map.pbtxt')

        with open(self._label_map_path) as f:
            label_map_string = f.read()
            label_map = StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)

        label_map_dict = {'__background__': 0}
        self.classes = ['__background__']

        for id, item in enumerate(label_map.item, 1):
            label_map_dict[item.name] = id
            self.classes.append(item.name)

        self.label_map_dict = label_map_dict

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(600),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

        with open(self._imgsetpath % self.image_set) as f:
            ids = f.readlines()

        self.ids = []
        for id in ids:
            striped_strings = id.strip().split()
            if len(striped_strings) == 2:
                self.ids.append(striped_strings[0])

    def __getitem__(self, index):
        img_id = self.ids[index]

        try:
            target = ET.parse(self._annopath % img_id).getroot()
            img = Image.open(self._imgpath % img_id).convert('RGB')
        except IOError as e:
            # logger.debug(e)
            return None

        origin_size = img.size

        img = self.transform(img)
        img = img.unsqueeze(0)
        target_size = tuple(img.size())
        im_info = np.array(
            [[float(target_size[2]), float(target_size[3]), 600. / min(origin_size)]])

        blobs = {}
        blobs['tensor'] = img
        blobs['im_info'] = im_info
        blobs['im_name'] = os.path.basename(self._imgpath % img_id)

        def bboxs(target):
            for obj in target.iter('object'):
                name = obj.find('name').text
                bb = obj.find('bndbox')
                bndbox = [bb.find('xmin').text, bb.find('ymin').text,
                          bb.find('xmax').text, bb.find('ymax').text]
                class_index = self.label_map_dict[name]
                yield bndbox, class_index

        try:
            gt_boxes, gt_classes = zip(*[box for box in bboxs(target)])
            gt_boxes = np.array(gt_boxes, dtype=np.uint16)
            gt_classes = np.array(gt_classes, dtype=np.int32)
        except ValueError as e:
            return None

        if self.target_transform is not None:
            target = self.target_transform(target)

        blobs['gt_classes'] = gt_classes
        blobs['boxes'] = gt_boxes * im_info[0][2]

        return blobs

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255, 0, 0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        img.show()


if __name__ == '__main__':
    root = '/data'
    ds = VOCDetection(root, 'train',
                      target_transform=TransformVOCDetectionAnnotation(False))
    print(len(ds))
    for d in ds:
        img = d

    print img
