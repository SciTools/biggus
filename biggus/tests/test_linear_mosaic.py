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
import unittest

import numpy as np

import biggus


class TestLinearMosaic(unittest.TestCase):
    def _tile(self, shape, dtype='f4'):
        data = np.arange(np.product(shape), dtype=dtype).reshape(shape)
        return biggus.NumpyArrayAdapter(data)

    def test_init(self):
        # Simple
        tile3x4 = self._tile((3, 4))
        mosaic = biggus.LinearMosaic(tile3x4, 0)
        mosaic = biggus.LinearMosaic([tile3x4], 0)

        # 2-dimensional tile array => error
        with self.assertRaises(ValueError):
            mosaic = biggus.LinearMosaic([[tile3x4], [tile3x4]], 0)

        # Different axis values
        mosaic = biggus.LinearMosaic([tile3x4, tile3x4], 0)
        mosaic = biggus.LinearMosaic([tile3x4, tile3x4], 1)
        with self.assertRaises(ValueError):
            mosaic = biggus.LinearMosaic([tile3x4, tile3x4], -1)
        with self.assertRaises(ValueError):
            mosaic = biggus.LinearMosaic([tile3x4, tile3x4], 2)

        # Tile shapes
        tile3x5 = self._tile((3, 5))
        with self.assertRaises(ValueError):
            mosaic = biggus.LinearMosaic([tile3x4, tile3x5], 0)
        mosaic = biggus.LinearMosaic([tile3x4, tile3x5], 1)

        tile5x4 = self._tile((5, 4))
        mosaic = biggus.LinearMosaic([tile3x4, tile5x4], 0)
        with self.assertRaises(ValueError):
            mosaic = biggus.LinearMosaic([tile3x4, tile5x4], 1)

        # dtypes
        tile3x4_i4 = self._tile((3, 4), 'i4')
        with self.assertRaises(ValueError):
            mosaic = biggus.LinearMosaic([tile3x4, tile3x4_i4], 0)

    def test_ndim(self):
        tile3x4 = self._tile((3, 4))
        mosaic = biggus.LinearMosaic(tile3x4, 0)
        self.assertEqual(mosaic.ndim, 2)

        mosaic = biggus.LinearMosaic([tile3x4, tile3x4], 0)
        self.assertEqual(mosaic.ndim, 2)

        mosaic = biggus.LinearMosaic([tile3x4, tile3x4], 1)
        self.assertEqual(mosaic.ndim, 2)

        tile3 = self._tile((3,))
        mosaic = biggus.LinearMosaic(tile3, 0)
        self.assertEqual(mosaic.ndim, 1)

        tile3x4x5 = self._tile((3, 4, 5))
        mosaic = biggus.LinearMosaic(tile3x4x5, 0)
        self.assertEqual(mosaic.ndim, 3)

    def test_dtype(self):
        for spec in ('f4', 'i2'):
            dtype = np.dtype(spec)
            mosaic = biggus.LinearMosaic(self._tile(6, dtype), 0)
            self.assertEqual(mosaic.dtype, dtype)

    def test_shape(self):
        tile3x4 = self._tile((3, 4))
        mosaic = biggus.LinearMosaic(tile3x4, 0)
        self.assertEqual(mosaic.shape, (3, 4))

        tile2x4 = self._tile((2, 4))
        mosaic = biggus.LinearMosaic([tile3x4, tile2x4], 0)
        self.assertEqual(mosaic.shape, (5, 4))

        tile3x5 = self._tile((3, 5))
        mosaic = biggus.LinearMosaic([tile3x4, tile3x5], 1)
        self.assertEqual(mosaic.shape, (3, 9))

    def test_getitem(self):
        tile3x4 = self._tile((3, 4))
        # NB. This is also testing ndarray(), so perhaps a different
        # name would be more accurate.
        mosaic = biggus.LinearMosaic(tile3x4, 0)

        result = mosaic[1].ndarray()
        target = [4, 5, 6, 7]
        np.testing.assert_array_equal(result, target)

        result = mosaic[:, 2].ndarray()
        target = [2, 6, 10]
        np.testing.assert_array_equal(result, target)

        result = mosaic[2, 1].ndarray()
        target = 9
        np.testing.assert_array_equal(result, target)

        mosaic = biggus.LinearMosaic([tile3x4, tile3x4], 1)

        result = mosaic[1].ndarray()
        target = [4, 5, 6, 7, 4, 5, 6, 7]
        np.testing.assert_array_equal(result, target)

        result = mosaic[1:2].ndarray()
        target = [[4, 5, 6, 7, 4, 5, 6, 7]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[0:2].ndarray()
        target = [[0, 1, 2, 3, 0, 1, 2, 3],
                  [4, 5, 6, 7, 4, 5, 6, 7]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[2:0:-1].ndarray()
        target = [[8, 9, 10, 11, 8, 9, 10, 11],
                  [4, 5, 6, 7, 4, 5, 6, 7]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[:, 5].ndarray()
        target = [1, 5, 9]
        np.testing.assert_array_equal(result, target)

        result = mosaic[:, 1::3].ndarray()
        target = [[1, 0, 3],
                  [5, 4, 7],
                  [9, 8, 11]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[::-2].ndarray()
        target = [[8, 9, 10, 11, 8, 9, 10, 11],
                  [0, 1, 2, 3, 0, 1, 2, 3]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[::-2, 1::3].ndarray()
        target = [[9, 8, 11],
                  [1, 0, 3]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[(0, 0, 2, 1), 1::3].ndarray()
        target = [[1, 0, 3],
                  [1, 0, 3],
                  [9, 8, 11],
                  [5, 4, 7]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[:, (3, 6, 1)].ndarray()
        target = [[3, 2, 1],
                  [7, 6, 5],
                  [11, 10, 9]]
        np.testing.assert_array_equal(result, target)

        result = mosaic[1, 6].ndarray()
        target = 6
        np.testing.assert_array_equal(result, target)

        result = mosaic[:, (0, 1, 3)]
        target = [[0,  1,  3],
                  [4,  5,  7],
                  [8,  9, 11]]

        np.testing.assert_array_equal(result, target)

        with self.assertRaises(TypeError):
            result = mosaic['foo'].ndarray()

    def test_getitem_adjust_axis(self):
        tile_1 = self._tile((3, 4, 5))
        tile_2 = self._tile((3, 4, 2))
        mosaic = biggus.LinearMosaic([tile_1, tile_2], 2)
        result = mosaic[1, :, :]
        self.assertEqual(result.shape, (4, 7))
        result = mosaic[:, 1, :]
        self.assertEqual(result.shape, (3, 7))

    def test_getitem_adjust_axis_2(self):
        tile_1 = self._tile((3, 4, 5))
        tile_2 = self._tile((3, 4, 2))
        mosaic = biggus.LinearMosaic([tile_1, tile_2], 2)
        result = mosaic[1, 1, :]
        self.assertEqual(result.shape, (7,))

    def test_ndarray(self):
        tile3x4 = self._tile((3, 4))
        mosaic = biggus.LinearMosaic(tile3x4, 0)
        result = mosaic.ndarray()
        target = [[0, 1, 2, 3],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11]]
        np.testing.assert_array_equal(result, target)

        tile2x4 = self._tile((2, 4))
        mosaic = biggus.LinearMosaic([tile3x4, tile2x4], 0)
        result = mosaic.ndarray()
        target = [[0, 1, 2, 3],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11],
                  [0, 1, 2, 3],
                  [4, 5, 6, 7]]
        np.testing.assert_array_equal(result, target)

        tile3x5 = self._tile((3, 5))
        mosaic = biggus.LinearMosaic([tile3x4, tile3x5], 1)
        result = mosaic.ndarray()
        target = [[0, 1, 2, 3, 0, 1, 2, 3, 4],
                  [4, 5, 6, 7, 5, 6, 7, 8, 9],
                  [8, 9, 10, 11, 10, 11, 12, 13, 14]]
        np.testing.assert_array_equal(result, target)

    def test__getitem_full_keys(self):
        full_keys = (np.array([0, 1]), slice(None, None, None))
        tile1 = self._tile((5, 11))
        tile2 = self._tile((5, 8))
        mosaic = biggus.LinearMosaic([tile1, tile2], 1)
        result = mosaic._getitem_full_keys(full_keys)
        expected = biggus.LinearMosaic([tile1, tile2], 1)
        result_array = result.ndarray()
        expected_array = expected.ndarray()[0:2, :]
        np.testing.assert_array_equal(result_array, expected_array)


if __name__ == '__main__':
    unittest.main()
