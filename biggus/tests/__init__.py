import numpy as np

try:
    from unittest import mock
except ImportError:
    import mock


class AccessCounter(object):
    """
    Something that acts like a NumPy ndarray, but which records how
    many times each element has been read.

    """
    def __init__(self, ndarray):
        self._ndarray = ndarray
        self.counts = np.zeros(ndarray.shape)

    @property
    def dtype(self):
        return self._ndarray.dtype

    @property
    def ndim(self):
        return self._ndarray.ndim

    @property
    def shape(self):
        return self._ndarray.shape

    def __array__(self):
        return self._ndarray

    def __getitem__(self, keys):
        self.counts[keys] += 1
        return self._ndarray[keys]

    def unique_counts(self):
        return set(np.unique(self.counts))


class _KeyGen(object):
    """Gives you the content of keys when indexing."""
    def __getitem__(self, keys):
        return keys

#: An object that can be indexed to return a usable key.
key_gen = _KeyGen()
