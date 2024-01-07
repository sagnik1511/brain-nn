import unittest


class TestDataModule(unittest.TestCase):
    def test_dataset(self):
        from brain.data import ListDataset

        sample_input_data_X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        sample_input_data_y = [1, 2, 3, 4, 5]
        ds = ListDataset(sample_input_data_X, sample_input_data_y)
        assert len(ds) == len(sample_input_data_y)
        X, y = ds[0]

    def test_loader(self):
        import numpy as np
        from brain.data import ListDataset, DataLoader

        sample_input_data_X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
        sample_input_data_y = [1, 2, 3, 4, 5, 6, 7]
        ds = ListDataset(sample_input_data_X, sample_input_data_y)
        batch_size = 2
        shuffle = True
        drop_last = True
        dl = DataLoader(ds, batch_size, shuffle, drop_last)
        for inp_x, inp_y in dl.load_data():
            assert (
                inp_x.shape[0] == batch_size and inp_y.shape[0] == batch_size
            ), "Batch size differed"
            assert isinstance(inp_x, np.ndarray) and isinstance(
                inp_y, np.ndarray
            ), "Different data tytpe found"


if __name__ == "__main__":
    unittest.main()
