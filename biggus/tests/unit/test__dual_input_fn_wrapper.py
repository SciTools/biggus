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
Unit tests for `biggus._dual_input_fn_wrapper` and the functions that it
has wrapped.

"""

import inspect
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import biggus


class Test__dual_input_fn_wrapper(unittest.TestCase):
    def test_docstring(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a + b)
        doc = inspect.getdoc(wrapped_fn)

        expected = ('Return the elementwise evaluation of '
                    'my_module.my_function(a, b) as another Array.')
        self.assertEqual(doc, expected)

    def test_auto_fn_name(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a + b)
        self.assertEqual(wrapped_fn.__name__, (lambda a, b: a + b).__name__)

    def test_given_fn_name(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a + b,
                                                   fn_name='identity')
        self.assertEqual(wrapped_fn.__name__, 'identity')

    def test_masked_array_creates_elementwise(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a + b,
                                                   lambda a, b: a - b,
                                                   fn_name='identity')
        result = wrapped_fn(np.array([-5, 2]), np.array([-5, 2]))
        self.assertIsInstance(result, biggus._Elementwise)

    def test_ndarray_expected_values(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a + b,
                                                   fn_name='identity')
        result = wrapped_fn(np.array([-5, 2]), np.array([-5, 2]))
        assert_array_equal(result.ndarray(), [-10, 4])

    def test_masked_array_expected_values(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a - b,
                                                   lambda a, b: a + b,
                                                   fn_name='identity')
        result = wrapped_fn(np.array([-5, 2]), np.array([-5, 2]))
        assert_array_equal(result.masked_array(), [-10, 4])

    def test_masked_array_undefined(self):
        wrapped_fn = biggus._dual_input_fn_wrapper('my_module.my_function',
                                                   lambda a, b: a + b,
                                                   fn_name='identity')
        msg = 'No <lambda> operation defined for masked arrays.'
        result = wrapped_fn(np.array([-5, 2]), np.array([-5, 2]))
        with self.assertRaisesRegexp(TypeError, msg):
            assert_array_equal(result.masked_array(), [0, 4])


class Test_wrapped_functions(unittest.TestCase):
    def test_all_fns(self):
        fns_to_test = ['copysign', 'nextafter', 'ldexp', 'fmod']
        arr1 = np.array([10, 2, 5])
        arr2 = np.array([1, 5, 15])
        biggus_arr1 = biggus.NumpyArrayAdapter(arr1)
        biggus_arr2 = biggus.NumpyArrayAdapter(arr2)

        for fn_name in fns_to_test:
            np_fn = getattr(np, fn_name)
            biggus_fn = getattr(biggus, fn_name)
            result = biggus_fn(biggus_arr1, biggus_arr2)
            assert_array_equal(result.ndarray(), np_fn(arr1, arr2))


if __name__ == '__main__':
    unittest.main()
