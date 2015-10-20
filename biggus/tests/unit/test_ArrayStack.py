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
"""Unit tests for `biggus.ArrayStack`."""

import copy
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from biggus import Array, ArrayStack, NumpyArrayAdapter, ConstantArray
from biggus.tests import mock


class Test___init___invalid(unittest.TestCase):
    def test_not_arrays(self):
        class BadArray(object):
            dtype = 'f'
            fill_value = 9
            shape = ()
        bad_arrays = np.array([BadArray()])
        with self.assertRaisesRegexp(ValueError, 'subclass'):
            ArrayStack(bad_arrays)


def fake_array(fill_value, dtype=np.dtype('f4')):
    return mock.Mock(shape=mock.sentinel.SHAPE, dtype=dtype,
                     fill_value=fill_value, spec=Array)


class Test___init___fill_values(unittest.TestCase):
    def test_nan_nan(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(np.nan)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertTrue(np.isnan(stack.fill_value))

    def test_nan_number(self):
        array1 = fake_array(np.nan)
        array2 = fake_array(42)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 1e+20)

    def test_number_nan(self):
        array1 = fake_array(42)
        array2 = fake_array(np.nan)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 1e+20)

    def test_number_number(self):
        array1 = fake_array(42)
        array2 = fake_array(42)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 42)

    def test_number_other_number(self):
        array1 = fake_array(42)
        array2 = fake_array(43)
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 1e+20)

    def test_matching_strings(self):
        array1 = fake_array('foo', np.dtype('S3'))
        array2 = fake_array('foo', np.dtype('S3'))
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 'foo')

    def test_different_strings(self):
        array1 = fake_array('foo', np.dtype('S3'))
        array2 = fake_array('bar', np.dtype('S3'))
        stack = ArrayStack(np.array([array1, array2]))
        self.assertEqual(stack.fill_value, 'N/A')


class Test_multidim_array_stack(unittest.TestCase):
    def setUp(self):
        self.arrays = [ConstantArray((), i) for i in range(6)]

    def test_stack_order_c_numpy_array_t1(self):
        # 1D stack of arrays shape (6,)
        res = ArrayStack.multidim_array_stack(self.arrays, (3, 2), order='C')
        arr = np.array([i for i in range(6)])
        target = np.reshape(arr, (3, 2), order='C')
        self.assertTrue(np.array_equal(res.ndarray(), target))

    def test_stack_order_c_numpy_array_t2(self):
        # 1D stack of arrays shape (6,)
        # alternate shape
        res = ArrayStack.multidim_array_stack(self.arrays, (2, 3), order='C')
        arr = np.array([i for i in range(6)])
        target = np.reshape(arr, (2, 3), order='C')
        self.assertTrue(np.array_equal(res.ndarray(), target))

    def test_stack_order_c_multidim(self):
        # 1D stack of 6 arrays each (4, 5)
        arrays = [NumpyArrayAdapter(np.arange(20).reshape(4, 5)) for
                  i in range(6)]
        res = ArrayStack.multidim_array_stack(arrays, (2, 3), order='C')
        arr = np.arange(20).reshape(4, 5) * np.ones((6, 4, 5))
        target = np.reshape(arr, (2, 3, 4, 5), order='C')
        self.assertTrue(np.array_equal(res.ndarray(), target))

    def test_stack_order_default(self):
        # 1D stack of arrays shape (6,)
        # Ensure that the default index ordering corresponds to C.
        res = ArrayStack.multidim_array_stack(self.arrays, (3, 2))
        arr = np.array([i for i in range(6)])
        target = np.reshape(arr, (3, 2), order='C')
        self.assertTrue(np.array_equal(res.ndarray(), target))

    def test_stack_order_fortran_t1(self):
        # Fortran index ordering
        # 1D stack of arrays shape (6,)
        res = ArrayStack.multidim_array_stack(self.arrays, (3, 2), order='F')
        arr = np.array([i for i in range(6)])
        target = np.reshape(arr, (3, 2), order='F')
        self.assertTrue(np.array_equal(res.ndarray(), target))

    def test_stack_order_fortran_t2(self):
        # Fortran index ordering
        # 1D stack of arrays shape (6,)
        # alternate shape
        res = ArrayStack.multidim_array_stack(self.arrays, (2, 3), order='F')
        arr = np.array([i for i in range(6)])
        target = np.reshape(arr, (2, 3), order='F')
        self.assertTrue(np.array_equal(res.ndarray(), target))

    def test_incompatible_shape(self):
        # 1D stack of arrays shape (6,)
        # Specifying a stack shape that is not feasible.
        msg = 'total size of new array must be unchanged'
        with self.assertRaisesRegexp(ValueError, msg):
            ArrayStack.multidim_array_stack(self.arrays, (3, 1), order='C')

    def test_multidim_stack_multidim(self):
        # Multidim stack of arrays shape (4, 6)
        arrays = [[ConstantArray((), i) for i in range(6)] for i in range(4)]
        msg = 'multidimensional stacks not yet supported'
        with self.assertRaisesRegexp(ValueError, msg):
            ArrayStack.multidim_array_stack(arrays, (3, 2, 4))

    def test_order_incorrect_order(self):
        # Specifying an unknown order.
        array1 = fake_array(0)
        array2 = fake_array(0)
        with self.assertRaisesRegexp(TypeError, 'order not understood'):
            ArrayStack.multidim_array_stack([array1, array2], (1, 2),
                                            order='random')


class Test___getitem__(unittest.TestCase):
    # Note, these are not a complete set of unit tests.
    # Currently they only handle the newaxis checking.
    # There are more tests in biggus.tests.test_stack.
    def setUp(self):
        self.a1 = ConstantArray([4, 3])
        self.a2 = ConstantArray([4, 3])
        self.a = ArrayStack([self.a1, self.a2])

    def test_newaxis_leading(self):
        self.assertEqual(self.a[np.newaxis].shape, (1, 2, 4, 3))

    def test_newaxis_trailing(self):
        self.assertEqual(self.a[..., np.newaxis].shape, (2, 4, 3, 1))


class Test__deepcopy__(unittest.TestCase):
    # Numpy <= 1.10 has a bug which prevents a deepcopy of an F-order
    # object array.
    # See https://github.com/SciTools/biggus/issues/157.
    def test_fortran_order(self):
        self.check('f')

    def test_c_order(self):
        self.check('c')

    def check(self, order):
        def adapt(value):
            return NumpyArrayAdapter(np.array(value))

        expected = np.array([[0, 1], [2, 3]], order=order)
        orig = ArrayStack(np.array([[adapt(0), adapt(1)],
                                    [adapt(2), adapt(3)]],
                                   order=order, dtype=object))
        copied = copy.deepcopy(orig)
        assert_array_equal(expected, copied.ndarray())


if __name__ == '__main__':
    unittest.main()
