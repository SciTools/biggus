# (C) British Crown Copyright 2015, Met Office
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
Unit tests for `biggus._ufunc_wrapper` and the functions that it
has wrapped.

"""

import inspect
import unittest
import warnings

import numpy as np
import numpy.ma
from numpy.testing import assert_array_equal
from numpy.ma.testutils import assert_array_equal as assert_masked_array_equal

import biggus


class Test__unary_fn_wrapper(unittest.TestCase):
    def test_nin2_docstring(self):
        wrapped_fn = biggus._ufunc_wrapper(np.add)
        doc = inspect.getdoc(wrapped_fn)

        expected = ('Return the elementwise evaluation of '
                    'np.add(a, b) as another Array.')
        self.assertEqual(doc, expected)

    def test_nin1_docstring(self):
        wrapped_fn = biggus._ufunc_wrapper(np.negative)
        doc = inspect.getdoc(wrapped_fn)

        expected = ('Return the elementwise evaluation of '
                    'np.negative(a) as another Array.')
        self.assertEqual(doc, expected)

    def test_non_ufunc(self):
        msg = 'not a ufunc'
        with self.assertRaisesRegexp(TypeError, msg):
            biggus._ufunc_wrapper(lambda x: x)

    def test_nout2_ufunc(self):
        msg = "Unsupported ufunc 'modf' with 1 input arrays & 2 output arrays."
        with self.assertRaisesRegexp(ValueError, msg):
            biggus._ufunc_wrapper(np.modf)


class Test_wrapped_functions(unittest.TestCase):
    def setUp(self):
        self.arr1 = np.array([1, 2, 3])
        self.arr2 = np.array([2, 1, 2])
        self.biggus_arr1 = biggus.NumpyArrayAdapter(self.arr1)
        self.biggus_arr2 = biggus.NumpyArrayAdapter(self.arr2)

        self.marr1 = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
        self.marr2 = np.ma.masked_array([1, 5, 2], mask=[0, 0, 1])
        self.biggus_marr1 = biggus.NumpyArrayAdapter(self.marr1)
        self.biggus_marr2 = biggus.NumpyArrayAdapter(self.marr2)

        ufunc_names = ['absolute', 'add', 'arccos', 'arccosh', 'arcsin',
                       'arcsinh', 'arctan', 'arctan2', 'arctanh',
                       'bitwise_and', 'bitwise_or', 'bitwise_xor',
                       'ceil', 'conj', 'cos', 'cosh', 'deg2rad', 'divide',
                       'equal', 'exp', 'exp2', 'expm1', 'floor',
                       'floor_divide', 'fmax', 'fmin', 'greater',
                       'greater_equal', 'hypot', 'invert', 'left_shift',
                       'less', 'less_equal', 'log',
                       'log10', 'log2', 'logical_and', 'logical_not',
                       'logical_or', 'logical_xor', 'maximum', 'minimum',
                       'multiply', 'negative', 'not_equal', 'power',
                       'rad2deg', 'reciprocal', 'right_shift', 'rint', 'sign',
                       'sin', 'sinh', 'sqrt', 'square', 'subtract', 'tan',
                       'tanh', 'true_divide', 'trunc']

        self.ufuncs = [(name, getattr(np, name), getattr(biggus, name))
                       for name in ufunc_names]

    def test_all_fns(self):
        self.ufuncs.append(('sub', np.subtract, biggus.subtract))

        for fn_name, ufunc, biggus_fn in self.ufuncs:
            if ufunc.nin == 1:
                result = biggus_fn(self.biggus_arr1)
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter("always")
                    expected = ufunc(self.arr1)
            else:
                result = biggus_fn(self.biggus_arr1, self.biggus_arr2)
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter("always")
                    expected = ufunc(self.arr1, self.arr2)

            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always")
                actual = result.ndarray()

            self.assertEqual([unicode(warning.message)
                              for warning in actual_warnings],
                             [unicode(warning.message)
                              for warning in expected_warnings])

            error_msg = ('biggus.{} produces different results to np.{}:'
                         ''.format(fn_name, fn_name))
            assert_array_equal(actual, expected, error_msg)

    def test_all_fns_masked(self):
        for fn_name, ufunc, biggus_fn in self.ufuncs:
            masked_ufunc = getattr(np.ma, ufunc.__name__, None)
            if masked_ufunc is not None:
                if ufunc.nin == 1:
                    result = biggus_fn(self.biggus_marr1)
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        expected = masked_ufunc(self.marr1)
                else:
                    result = biggus_fn(self.biggus_marr1, self.biggus_marr2)
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        expected = masked_ufunc(self.marr1, self.marr2)

                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    actual = result.masked_array()

                error_msg = ('biggus.{} produces different results to '
                             'np.ma.{}'.format(fn_name, ufunc.__name__))
                assert_masked_array_equal(actual, expected, error_msg)


if __name__ == '__main__':
    unittest.main()
