# (C) British Crown Copyright 2014 - 2015, Met Office
#
# This file is part of Biggus.
#
# Biggus is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Biggus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Biggus. If not, see <http://www.gnu.org/licenses/>.

from contextlib import contextmanager

try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np

import biggus


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


@contextmanager
def set_chunk_size(value):
    old_chunk_size = biggus.MAX_CHUNK_SIZE
    biggus.MAX_CHUNK_SIZE = value
    yield
    biggus.MAX_CHUNK_SIZE = old_chunk_size
