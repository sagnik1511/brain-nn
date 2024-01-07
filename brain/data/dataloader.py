"""
This module holds all the code for preparation of a loader
"""
import random
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.ds = dataset
        self.bs = batch_size
        self.shuffle = shuffle
        self.iterable_count = (
            (len(self.ds) // self.bs) * self.bs if drop_last else len(self.ds)
        )
        self.order = [idx for idx in range(len(self.ds))]
        self.obj_count = None
        self._test_sanity()

    def load_data(self):
        if self.shuffle:
            random.shuffle(self.order)
        batched_data = []
        for idx in range(self.iterable_count):
            batched_data.append(self.ds[self.order[idx]])
            if (idx + 1) % self.bs == 0 or idx == self.iterable_count - 1:
                yield self._compile_batch(batched_data)
                batched_data = []

    def _compile_batch(self, batch):
        """
        The function converts tuple of records to tuple of batches
        """
        output_batch = []
        for el in range(self.obj_count):
            obj = np.array([data[el] for data in batch])
            output_batch.append(obj)

        return tuple(output_batch)

    def _test_sanity(self):
        for idx in range(len(self.ds)):
            op = self.ds[idx]
            if not self.obj_count:
                self.obj_count = len(op)
            assert (
                len(op) == self.obj_count
            ), f"DataLoader encountered a record with {len(op)} parts, expected parts {self.obj_count}"
