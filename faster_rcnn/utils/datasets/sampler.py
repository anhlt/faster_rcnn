from torch.utils.data.sampler import Sampler


__all__ = ["SortedIndexSampler"]

class SortedIndexSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, sorted_index):
        self.sorted_index = sorted_index

    def __iter__(self):
        return iter(self.sorted_index)

    def __len__(self):
        return len(self.sorted_index)
