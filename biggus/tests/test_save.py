# (C) British Crown Copyright 2013, Met Office
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
"""
Unit tests for `biggus.save()`.

"""

import unittest

import numpy as np
import numpy.ma

import biggus


class _WriteCounter(object):
    """
    Acts like an HDF5 or netCDF4 variable, but records which slices
    have been written.

    NB. Assumes all write attempts will access the entirety of the
    last dimension.

    """
    def __init__(self, shape):
        self._written = np.zeros(shape[:-1], dtype=np.bool)

    def __setitem__(self, keys, values):
        assert keys[-1] == slice(None)
        self._written[keys[:-1]] = True

    def all_written(self):
        return np.all(self._written)


class TestWritePattern(unittest.TestCase):
    # Check the save operation writes to all of the expected areas
    # of the target.
    def _small_array(self):
        shape = (768, 1024)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        array = biggus.NumpyArrayAdapter(data)
        return array

    def test_small(self):
        # Source data: 3 MB
        array = self._small_array()
        target = _WriteCounter(array.shape)
        biggus.save([array], [target])
        self.assertTrue(target.all_written())

    def test_large(self):
        # Source data: 15 GB
        small_array = self._small_array()
        array = biggus.ArrayStack([[small_array] * 1000] * 5)
        target = _WriteCounter(array.shape)
        biggus.save([array], [target])
        self.assertTrue(target.all_written())


class TestMaskedSave(unittest.TestCase):
    # check that the masked keyword arguement puts a masked array in the target
    def _small_array(self):
        shape = (768, 1024)
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        data = np.ma.array(data, mask=False)
        return data

    def _masked_array_with_mask(self):
        array = self._small_array()
        array.mask[0, 7] = True
        array = biggus.NumpyArrayAdapter(array)
        return array

    def _masked_array(self):
        array = self._small_array()
        return array

    def test_mask(self):
        source = self._masked_array_with_mask()
        target = self._masked_array()
        biggus.save([source], [target], masked=True)
        self.assertTrue(target.mask[0, 7])


class TestNumbers(unittest.TestCase):
    # Check the numeric results of the save operation.
    def test_numbers(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4) + 10
        array = biggus.NumpyArrayAdapter(data)
        target = np.zeros((3, 4))
        biggus.save([array], [target])
        np.testing.assert_array_equal(data, target)


if __name__ == '__main__':
    unittest.main()
