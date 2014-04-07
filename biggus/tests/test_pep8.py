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
Perform a PEP8 conformance test of the Biggus code base.

"""
import os
import unittest

import pep8

import biggus


class TestCodeFormat(unittest.TestCase):
    """Test code conformance."""

    def test_pep8_conformance(self):
        # Tests the biggus code base against the "pep8" tool.
        #
        # Users can add their own excluded files (should files exist in the
        # local directory which is not in the repository) by adding a
        # ".pep8_test_exclude.txt" file in the same directory as this test.
        # The file should be a line separated list of filenames/directories
        # as can be passed to the "pep8" tool's exclude list.

        pep8style = pep8.StyleGuide(quiet=False)
        pep8style.options.exclude.extend([])

        # Allow users to add their own exclude list.
        test_root = os.path.abspath(__file__)
        extra_exclude_file = os.path.join(os.path.dirname(test_root),
                                          '.pep8_test_exclude.txt')
        if os.path.exists(extra_exclude_file):
            with open(extra_exclude_file, 'r') as fhandle:
                extra_exclude = [line.strip()
                                 for line in fhandle if line.strip()]
            pep8style.options.exclude.extend(extra_exclude)

        root = os.path.abspath(biggus.__file__)
        result = pep8style.check_files([os.path.dirname(root)])
        self.assertEqual(result.total_errors, 0, "Found code syntax "
                                                 "errors (and warnings).")


if __name__ == '__main__':
    unittest.main()
