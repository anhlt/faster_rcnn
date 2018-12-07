import torch.utils.data as data
import os
import os.path
from .voc.voc import VOCDetection
try:
    from .voc.string_int_label_map_pb2 import StringIntLabelMap
except Exception as e:
    from voc.string_int_label_map_pb2 import StringIntLabelMap

from google.protobuf import text_format


class VOCMerge(data.Dataset):
    def __init__(self, root, image_set, dataset_name='merge_voc', *args, **kwargs):
        self.root = root
        self.image_set = image_set
        self._label_map_path = os.path.join(
            self.root, dataset_name, 'pascal_label_map.pbtxt')
        self.sub_class_dict = {}

        with open(self._label_map_path) as f:
            label_map_string = f.read()
            label_map = StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)

        label_map_dict = {'__background__': 0}
        self.classes = ['__background__']
        self.sum_item = [0]

        for id, item in enumerate(label_map.item, 1):
            label_map_dict[item.name] = id
            self.classes.append(item.name)
        self.label_map_dict = label_map_dict

        for sub_class in self.classes[1:]:
            self.sub_class_dict[sub_class] = VOCDetection(
                os.path.join(root, dataset_name), sub_class + '_' + image_set,
                dataset_name=sub_class + "_output",
                *args, **kwargs)

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
