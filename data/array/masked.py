import numpy as np

from crossfit.data.array.dispatch import crossarray


class MaskedArray:
    def __init__(self, data, mask=None):
        with crossarray:
            self.data = data
            if mask is None:
                self.mask = np.zeros_like(self.data, dtype=np.bool_)
            else:
                self.mask = np.array(mask)

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
