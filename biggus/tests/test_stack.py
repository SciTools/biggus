# (C) British Crown Copyright 2012 - 2015, Met Office
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


class TestStack(unittest.TestCase):
    def test_dtype(self):
        dtype = np.dtype('f4')
        item = biggus.NumpyArrayAdapter(np.empty(6, dtype=dtype))
        stack = np.array([item], dtype='O')
        array = biggus.ArrayStack(stack)
        self.assertEqual(array.dtype, dtype)

    def test_shape_and_getitem(self):
        # Sequence of tests, defined as:
        #   1. Stack shape,
        #   2. item shape,
        #   3. sequence of indexing operations to apply,
        #   4. expected result shape or exception.
        tests = [
            [(6, 70), (30, 40), [], (6, 70, 30, 40)],
            [(6, 70), (30, 40), [5], (70, 30, 40,)],
            [(6, 70), (30, 40), [(5,)], (70, 30, 40,)],
            [(6, 70), (30, 40), [5, 3], (30, 40)],
            [(6, 70), (30, 40), [(5, 3)], (30, 40)],
            [(6, 70), (30, 40), [5, 3, 2], (40,)],
            [(6, 70), (30, 40), [(5, 3, 2)], (40,)],
            [(6, 70), (30, 40), [5, 3, 2, 1], ()],
            [(6, 70), (30, 40), [(5, 3, 2, 1)], ()],
            [(6, 70), (30, 40), [5, (3, 2), 1], ()],
            [(6, 70), (30, 40), [(slice(None, None), 6)], (6, 30, 40)],
            [(6, 70), (30, 40), [(slice(None, None), np.newaxis, 6)],
             (6, 1, 30, 40)],
            [(6, 70), (30, 40), [(slice(None, None), slice(1, 5))],
             (6, 4, 30, 40)],
            [(6, 70), (30, 40), [(slice(None, None),), 4], (70, 30, 40,)],
            [(6, 70), (30, 40), [5, (slice(None, None),)], (70, 30, 40,)],
            [(6, 70), (30, 40), [(slice(None, 10),)], (6, 70, 30, 40)],
            [(6, 70), (30, 40), [(slice(None, 10),), 5], (70, 30, 40,)],
            [(6, 70), (30, 40), [(slice(None, 10),), (slice(None, 3),)],
             (3, 70, 30, 40)],
            [(6, 70), (30, 40), [(slice(None, 10),), (slice(None, None, 2),)],
             (3, 70, 30, 40)],
            [(6, 70), (30, 40), [(slice(5, 10),),
                                 (slice(None, None), slice(2, 6))],
             (1, 4, 30, 40)],
            [(6, 70), (30, 40), [(slice(None, None), slice(2, 6)),
                                 (slice(5, 10),)], (1, 4, 30, 40)],
            [(6, 70), (30, 40), [3.5], ValueError],
            [(6, 70), (30, 40), ['foo'], ValueError],
            [(6, 70), (30, 40), [object()], ValueError],
        ]
        dtype = np.dtype('f4')
        for stack_shape, item_shape, cuts, target in tests:

            def make_array(*n):
                concrete = np.empty(item_shape, dtype)
                array = biggus.NumpyArrayAdapter(concrete)
                return array

            stack = np.empty(stack_shape, dtype='O')
            for index in np.ndindex(stack.shape):
                stack[index] = make_array()
            array = biggus.ArrayStack(stack)
            if isinstance(target, type):
                with self.assertRaises(target):
                    for cut in cuts:
                        array = array.__getitem__(cut)
            else:
                for cut in cuts:
                    array = array.__getitem__(cut)
                    self.assertIsInstance(array, biggus.Array)
                self.assertEqual(array.shape, target,
                                 '\nCuts: {!r}'.format(cuts))

    def test_ndarray(self):
        # Sequence of tests, defined as:
        #   1. Stack shape,
        #   2. item shape,
        #   3. expected result.
        tests = [
            [(1,), (3,), np.arange(3).reshape(1, 3)],
            [(1,), (3, 4), np.arange(12).reshape(1, 3, 4)],
            [(6,), (3, 4), np.arange(72).reshape(6, 3, 4)],
            [(6, 70), (3, 4), np.arange(5040).reshape(6, 70, 3, 4)],
        ]
        for stack_shape, item_shape, target in tests:
            stack = np.empty(stack_shape, dtype='O')
            item_size = np.array(item_shape).prod()
            for index in np.ndindex(stack.shape):
                start = np.ravel_multi_index(index, stack_shape) * item_size
                concrete = np.arange(item_size).reshape(item_shape)
                concrete += start
                array = biggus.NumpyArrayAdapter(concrete)
                stack[index] = array
            array = biggus.ArrayStack(stack)
            result = array.ndarray()
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(array.dtype, result.dtype)
            self.assertEqual(array.shape, result.shape)
            np.testing.assert_array_equal(result, target)


if __name__ == '__main__':
    unittest.main()
