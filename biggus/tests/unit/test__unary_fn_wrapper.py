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
Unit tests for `biggus._unary_fn_wrapper` and the functions that it
has wrapped.

"""

import inspect
import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import biggus


class Test__unary_fn_wrapper(unittest.TestCase):
    # Attach a future proof assert raises method.
    if sys.version_info[0] == 2:
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    def test_docstring(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a)
        doc = inspect.getdoc(wrapped_fn)

        expected = ('Return the elementwise evaluation of '
                    'my_module.my_function(a) as another Array.')
        self.assertEqual(doc, expected)

    def test_auto_fn_name(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a)
        self.assertEqual(wrapped_fn.__name__, (lambda a: a).__name__)

    def test_given_fn_name(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a, fn_name='identity')
        self.assertEqual(wrapped_fn.__name__, 'identity')

    def test_masked_array_creates_elementwise(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a + 10,
                                              lambda a: a - 10,
                                              fn_name='identity')
        self.assertIsInstance(wrapped_fn(np.array([1])), biggus._Elementwise)

    def test_unexpected_n_arguments(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a + 10,
                                              fn_name='identity')
        # TODO: It would be good if this were not called "wrapped_function".
        if sys.version_info[0] == 2:
            msg = 'wrapped_function\(\) takes exactly 1 argument \(2 given\)'
        else:
            msg = ('wrapped_function\(\) takes 1 positional argument but 2 '
                   'were given')
        with self.assertRaisesRegex(TypeError, msg) as ex:
            wrapped_fn(np.array([-5, 2]), np.array([-5, 2]))

    def test_ndarray_expected_values(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a + 10,
                                              fn_name='identity')
        assert_array_equal(wrapped_fn(np.array([-5, 2])).ndarray(),
                           [5, 12])

    def test_masked_array_expected_values(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a + 10,
                                              lambda a: a - 10,
                                              fn_name='identity')
        assert_array_equal(wrapped_fn(np.array([-5, 2])).masked_array(),
                           [-15, -8])

    def test_masked_array_undefined(self):
        wrapped_fn = biggus._unary_fn_wrapper('my_module.my_function',
                                              lambda a: a + 10,
                                              fn_name='identity')
        msg = 'No <lambda> operation defined for masked arrays.'
        with self.assertRaisesRegex(TypeError, msg):
            assert_array_equal(wrapped_fn(np.array([-5, 2])).masked_array(),
                               [-15, -8])


class Test_wrapped_functions(unittest.TestCase):
    def test_all_fns(self):
        fns_to_test = ['isreal', 'iscomplex', 'isinf', 'isnan', 'signbit']
        arr = np.array([-10, 0, 5])
        biggus_arr = biggus.NumpyArrayAdapter(arr)
        for fn_name in fns_to_test:
            np_fn = getattr(np, fn_name)
            biggus_fn = getattr(biggus, fn_name)
            result = biggus_fn(biggus_arr)
            assert_array_equal(result.ndarray(), np_fn(arr))


if __name__ == '__main__':
    unittest.main()
