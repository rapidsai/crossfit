import numpy as np

from crossfit.data.array.dispatch import crossarray


class MaskedArray:
    def __init__(self, data, mask=None):
        with crossarray:
            self.data = data
            if mask is None:
                self.mask = np.zeros_like(self.data, dtype=np.bool_)
            else:
                self.mask = mask

    def __getitem__(self, index):
        new_data = self.data[index]
        new_mask = self.mask[index]
        return MaskedArray(new_data, new_mask)

    def __setitem__(self, index, value):
        if isinstance(value, MaskedArray):
            self.data[index] = value.data
            self.mask[index] = value.mask
        else:
            self.data[index] = value
            self.mask[index] = False  # Assuming you want to unmask this value

    def __add__(self, other):
        if isinstance(other, MaskedArray):
            new_data = self.data + other.data
            new_mask = self.mask | other.mask
        else:
            new_data = self.data + other
            new_mask = self.mask

        return MaskedArray(new_data, new_mask)

    def sum(self):
        with crossarray:
            return np.sum(self.data[~self.mask])

    def filled(self, fill_value):
        with crossarray:
            output_data = np.copy(self.data)
            output_data[self.mask] = fill_value
            return output_data

    @property
    def shape(self):
        return self.data.shape
