from __future__ import division
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
        self._annopath = os.path.join(
            self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

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
    def __init__(self, root, image_set,
                 dataset_name='VOC2007', general_transform=None,
                 transform=None, target_transform=None, oversample=False, oversample_len=100, read_img=True):
        self.root = root
        self.image_set = image_set
        self.target_transform = target_transform
        self.dataset_name = dataset_name
        self._annopath = os.path.join(
            self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')
        self._label_map_path = os.path.join(
            self.root, dataset_name, 'pascal_label_map.pbtxt')
        self.oversample = oversample
        self.general_transform = general_transform
        self.oversample_len = oversample_len
        self.read_img = read_img


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
        img_id = self.ids[index % len(self.ids)]

        try:
            target = ET.parse(self._annopath % img_id).getroot()
        except IOError as e:
            logger.debug(e)
            logger.debug(self._annopath % img_id)
            return None

        origin_size = (int(target.find('size').find('width').text),
                       int(target.find('size').find('height').text))
        def bboxs(target, origin_size):
            for obj in target.iter('object'):
                name = obj.find('name').text
                bb = obj.find('bndbox')
                bndbox = [float(bb.find('xmin').text), float(bb.find('ymin').text),
                          float(bb.find('xmax').text), float(bb.find('ymax').text)]

                bndbox[0] = max(0, bndbox[0])
                bndbox[1] = max(0, bndbox[1])
                bndbox[2] = min(origin_size[0], bndbox[2])
                bndbox[3] = min(origin_size[1], bndbox[3])

                if 0. <= bndbox[0] < bndbox[2] <= origin_size[0] and 0. <= bndbox[1] < bndbox[3] <= origin_size[1]:
                    class_index = self.label_map_dict[name]
                    yield bndbox, class_index
                else:
                    print bndbox, origin_size, self._annopath % img_id

        try:
            gt_boxes, gt_classes = zip(
                *[box for box in bboxs(target, origin_size)])
            gt_boxes = np.array(gt_boxes, dtype=np.int32)
            gt_classes = np.array(gt_classes, dtype=np.int32)
        except ValueError as e:
            logger.debug(e)
            return None
        except AssertionError as e:
            logger.debug(e)
            return None

        if self.read_img:
            if self.target_transform is not None:
                target = self.target_transform(target)

            img = Image.open(self._imgpath % img_id).convert('RGB')
            if self.general_transform:
                img, gt_boxes = self.general_transform(img, gt_boxes)
            blobs = {}
            blobs['gt_classes'] = gt_classes
            # blobs['boxes'] = gt_boxes * im_info[0][2]
            # blobs['tensor'] = img
            blobs['boxes'] = gt_boxes
            blobs['tensor'] = self.transform(img).unsqueeze(0)

            target_size = tuple(img.size)
            im_info = np.array(
                [[float(target_size[0]), float(target_size[1]), min(target_size) / min(origin_size)]])

            blobs['im_info'] = im_info
            blobs['im_name'] = os.path.basename(self._imgpath % img_id)

            return blobs
        else:
            blobs = {}
            blobs['im_name'] = os.path.basename(self._imgpath % img_id)
            blobs['origin_size'] = origin_size
            return blobs

    def __len__(self):
        if self.oversample:
            return max(self.oversample_len, len(self.ids))
        else:
            return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255, 0, 0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        img.show()

    def clean(self, commit=False, name=None):
        annotation_dir = (os.path.join(
            self.root, self.dataset_name, 'Annotations'))
        img_dir = (os.path.join(self.root, self.dataset_name, 'JPEGImages'))
        dataset_dir = os.path.join(
            self.root, self.dataset_name, 'ImageSets', 'Main')
        train_file = os.path.join(dataset_dir, "%s_train.txt" % name)
        val_file = os.path.join(dataset_dir, "%s_val.txt" % name)

        annotation_files = [file for file in os.listdir(annotation_dir)]
        image_files = [file for file in os.listdir(img_dir)]

        need_delete_files = []
        retain_files = []

        for file in annotation_files:
            target = ET.parse(os.path.join(annotation_dir, file)).getroot()
            file_name = target.find("filename").text
            if not target.find('object'):
                need_delete_files.append(os.path.join(annotation_dir, file))
                need_delete_files.append(
                    os.path.join(img_dir, "%s.jpg" % file_name))

            else:
                retain_files.append(file_name)

            if not os.path.isfile(os.path.join(img_dir, "%s.jpg" % file_name)):
                print('Image not existed')
                need_delete_files.append(os.path.join(annotation_dir, file))    
                
        print "Need to delete: %d" % len(need_delete_files)
        print "Retain: %d" % len(retain_files)
        print "dataset_dir: %s" % dataset_dir

        print "train_file: %s" % train_file
        print "train_file: %s" % val_file
        last_train_index = int(len(retain_files) / 5 * 4)
        if commit:
            for file in need_delete_files:
                try:
                    os.remove(file)
                except OSError:
                    print ("File %s Not Existed!" % file)
        with open(train_file, 'w') as f:
            for file_name in retain_files[:last_train_index]:
                if commit:
                    f.write("%s 1\n" % file_name)
                else:
                    print("%s 1\n" % file_name)

        with open(val_file, 'w') as f:
            for file_name in retain_files[last_train_index:]:
                if commit:
                    f.write("%s 1\n" % file_name)
                else:
                    print("%s 1\n" % file_name)


if __name__ == '__main__':
    root = '/data'
    ds = VOCDetection(root, 'train',
                      target_transform=TransformVOCDetectionAnnotation(False))
    print(len(ds))
    for d in ds:
        img = d

    print img
