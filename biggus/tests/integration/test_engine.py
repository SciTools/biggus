# (C) British Crown Copyright 2014, Met Office
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
"""Integration tests for swappable engine."""

import unittest

import mock

import biggus


class Test(unittest.TestCase):
    def test(self):
        # If we switch evaluation engine, does it get used?
        chunk_handler = mock.Mock(name='chunk_handler')
        chunk_handler_class = mock.Mock(return_value=chunk_handler)
        array = biggus._Aggregation(mock.sentinel.array, mock.sentinel.axis,
                                    chunk_handler_class, {})
        engine = mock.Mock()
        default_engine = biggus.engine
        try:
            biggus.engine = engine
            array.ndarray()
            engine.process_chunks.assert_called_once_with(
                mock.sentinel.array, chunk_handler.add_chunk, False)
        finally:
            biggus.engine = default_engine


if __name__ == '__main__':
    unittest.main()
