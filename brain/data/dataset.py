"""
This module holds code for Dataset wrapper class
"""


class Dataset:
    def __len__(self):
        """Length / number of records of the dataset"""
        pass

    def __getitem__(self, index):
        """Fetching mechanism of each records to prepare a trainable input"""
        pass


class ListDataset(Dataset):
    """
    Simplest Dataset class to parse list data
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        length = len(self.X)
        assert length > 0, "Dataset must have atleast one record"
        return length

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
