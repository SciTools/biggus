# (C) British Crown Copyright 2016, Met Office
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
Virtual arrays of arbitrary size, with arithmetic and statistical
operations, and conversion to NumPy ndarrays.

Virtual arrays can be stacked to increase their dimensionality, or
tiled to increase their extent.

Includes support for easily wrapping data sources which produce NumPy
ndarray objects via slicing. For example: netcdf4python Variable
instances, and NumPy ndarray instances.

All operations are performed in a lazy fashion to avoid overloading
system resources. Conversion to a concrete NumPy ndarray requires an
explicit method call.

For example::

    # Wrap two large data sources (e.g. 52000 x 800 x 600).
    measured = OrthoArrayAdapter(netcdf_var_a)
    predicted = OrthoArrayAdapter(netcdf_var_b)

    # No actual calculations are performed here.
    error = predicted - measured

    # *Appear* to calculate the mean over the first dimension, and
    # return a new biggus Array with the correct shape, etc.
    # NB. No data are read and no calculations are performed.
    mean_error = biggus.mean(error, axis=0)

    # *Actually* calculate the mean, and return a NumPy ndarray.
    # This is when the data are read, subtracted, and the mean derived,
    # but all in a chunk-by-chunk fashion which avoids using much
    # memory.
    mean_error = mean_error.ndarray()

"""

from __future__ import absolute_import, division, print_function
from six.moves import (filter, input, map, range, zip)  # noqa

from ._init import *


__all__ = _init.__all__


__version__ = '0.14.0'


engine = AllThreadedEngine()
"""
The current lazy evaluation engine.

Defaults to an instance of :class:`AllThreadedEngine`.

"""
