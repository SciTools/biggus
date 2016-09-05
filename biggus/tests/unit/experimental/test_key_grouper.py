# (C) British Crown Copyright 2017, Met Office
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
"""Unit tests for `biggus.key_grouper`."""

from __future__ import absolute_import, division, print_function
from six.moves import (filter, input, map, range, zip)  # noqa

import unittest

from biggus.experimental.key_grouper import (
        dimension_group_to_lowest_common, normalize_slice, group_keys)


class Test_normalize_slice(unittest.TestCase):
    def assertSlice(self, expected_start, expected_stop, expected_step,
                    result):
        self.assertEqual(expected_start, result.start)
        self.assertEqual(expected_stop, result.stop)
        self.assertEqual(expected_step, result.step)

    def test_step_1(self):
        r = normalize_slice(slice(None, None, 1), 3)
        self.assertSlice(None, None, None, r)

    def test_step_m1(self):
        r = normalize_slice(slice(None, None, None), None)
        self.assertSlice(None, None, None, r)

    def test_stop_m1(self):
        r = normalize_slice(slice(None, -1), 10)
        self.assertSlice(None, 9, None, r)

    def test_start_m1(self):
        r = normalize_slice(slice(-2, -1), 10)
        self.assertSlice(8, 9, None, r)

    def test_start_0(self):
        r = normalize_slice(slice(0, None), 10)
        self.assertSlice(None, None, None, r)

    def test_reverse_negative(self):
        r = normalize_slice(slice(-1, -3, -1), 10)
        self.assertSlice(9, 7, -1, r)

    def test_reverse_negative_nothing_there(self):
        r = normalize_slice(slice(-3, -1, -1), 10)
        # Would actually result in no index, but it is still valid.
        self.assertSlice(7, 9, -1, r)


class Test_dim_grouper(unittest.TestCase):
    def setUp(self):
        class Foo(object):
            def __getitem__(self, keys):
                return keys
        self.indexer = Foo()

    def test_one_set(self):
        r = dimension_group_to_lowest_common(4, [[self.indexer[0:]]])
        e = {(None, None, None): [[self.indexer[:]]]}
        self.assertEqual(e, r)

    def test_identical(self):
        r = dimension_group_to_lowest_common(4, [[self.indexer[:]],
                                                 [self.indexer[:]]])
        e = {(None, None, None): [[self.indexer[:]], [self.indexer[:]]]}
        self.assertEqual(e, r)

    def test_repeat(self):
        r = dimension_group_to_lowest_common(4, [[self.indexer[:],
                                                  self.indexer[:]],
                                                 [self.indexer[:]]])
        e = {(None, None, None): [[self.indexer[:], self.indexer[:]],
                                  [self.indexer[:]]]}
        self.assertEqual(e, r)

    def test_equal(self):
        r = dimension_group_to_lowest_common(4, [[self.indexer[0:]],
                                                 [self.indexer[::1]]])
        e = {(None, None, None): [[self.indexer[:]], [self.indexer[:]]]}
        self.assertEqual(e, r)

    def test_single_subset(self):
        r = dimension_group_to_lowest_common(4, [[self.indexer[:2],
                                                  self.indexer[2:]],
                                                 [self.indexer[:]]])
        e = {(None, None, None): [[self.indexer[:2], self.indexer[2:]],
                                  [self.indexer[:]]]}
        self.assertEqual(e, r)

    def test_multiple_offset_subsets(self):
        indices = [[self.indexer[:2], self.indexer[2:4], self.indexer[4:7]],
                   [self.indexer[:4], self.indexer[4:5], self.indexer[5:]]]
        r = dimension_group_to_lowest_common(7, indices)
        e = {(None, 4, None): [[self.indexer[:2], self.indexer[2:4]],
                               [self.indexer[:4]]],
             (4, None, None): [[self.indexer[4:]],
                               [self.indexer[4:5], self.indexer[5:]]]
             }
        self.assertEqual(e, r)

    def test_unordered(self):
        # As test_multiple_offset_subsets, but with the input order switched.
        indices = [[self.indexer[2:4], self.indexer[:2], self.indexer[4:7]],
                   [self.indexer[:4], self.indexer[4:5], self.indexer[5:]]]
        r = dimension_group_to_lowest_common(7, indices)
        e = {(None, 4, None): [[self.indexer[:2], self.indexer[2:4]],
                               [self.indexer[:4]]],
             (4, None, None): [[self.indexer[4:]],
                               [self.indexer[4:5], self.indexer[5:]]]
             }
        self.assertEqual(e, r)


class Test_group_keys(unittest.TestCase):
    def setUp(self):
        class Foo(object):
            def __getitem__(self, keys):
                return keys
        self.ind = Foo()
        self.maxDiff = None
        self.sorted_keys = lambda dkeys: sorted(dkeys.keys(),
                                                key=lambda key: str(key))

    def test_one_group(self):
        colon = self.ind[:]
        r = group_keys((6, 2),
                       [(self.ind[:3], colon), (self.ind[3:], colon)],
                       [(colon, colon)])
        e = {((None, None, None), (None, None, None)): (
                [(self.ind[:3], colon), (self.ind[3:], colon)],
                [(colon, colon)])
             }
        self.assertEqual(e, r)

    def test_two_groups_same_dim(self):
        colon = self.ind[:]
        r = group_keys((6, 2),
                       [(self.ind[:2], colon), (self.ind[2:4], colon),
                        (self.ind[4:], colon)],
                       [(self.ind[:4], colon), (self.ind[4:], colon)])
        e = {((None, 4, None), (None, None, None)): ([(self.ind[:2], colon),
                                                      (self.ind[2:4], colon)],
                                                     [(self.ind[:4], colon)]),
             ((4, None, None), (None, None, None)): ([(self.ind[4:], colon)],
                                                     [(self.ind[4:], colon)]),
             }
        self.assertEqual(self.sorted_keys(e), self.sorted_keys(r))
        self.assertEqual(e, r)

    def test_one_input(self):
        # One input always results in n keys groups (because they naturally
        # split perfectly).
        colon = self.ind[:]
        r = group_keys((6, 2),
                       [(self.ind[:2], colon), (self.ind[2:4], colon),
                        (self.ind[4:], colon)])
        e = {((None, 2, None), (None, None, None)): ([(self.ind[:2], colon)],),
             ((2, 4, None), (None, None, None)): ([(self.ind[2:4], colon)],),
             ((4, None, None), (None, None, None)): ([(self.ind[4:], colon)],)}
        self.assertEqual(self.sorted_keys(e), self.sorted_keys(r))
        self.assertEqual(e, r)

    def test_one_input_one_dim(self):
        r = group_keys((6,), [(slice(0, 5, None),), (slice(5, 6, None),)])
        e = {((None, 5, None),): ([(slice(0, 5, None),)],),
             ((5, None, None),): ([(slice(5, 6, None), )],)
             }
        self.assertEqual(self.sorted_keys(e), self.sorted_keys(r))
        self.assertEqual(e, r)


if __name__ == '__main__':
    unittest.main()
