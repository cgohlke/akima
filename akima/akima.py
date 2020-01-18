# akima.py

# Copyright (c) 2007-2020, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Akima Interpolation.

Akima is a Python library that implements Akima's interpolation method
described in:

    A new method of interpolation and smooth curve fitting based
    on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.

A continuously differentiable sub-spline is built from piecewise cubic
polynomials. It passes through the given data points and will appear smooth
and natural.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.1.1

Requirements
------------
* `CPython >= 3.6 <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_

Notes
-----
The Akima module is no longer being actively developed.

Consider using `scipy.interpolate.Akima1DInterpolator
<http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ instead.

Examples
--------
>>> from matplotlib import pyplot
>>> from scipy.interpolate import Akima1DInterpolator
>>> def example():
...     '''Plot interpolated Gaussian noise.'''
...     x = numpy.sort(numpy.random.random(10) * 100)
...     y = numpy.random.normal(0.0, 0.1, size=len(x))
...     x2 = numpy.arange(x[0], x[-1], 0.05)
...     y2 = interpolate(x, y, x2)
...     y3 = Akima1DInterpolator(x, y)(x2)
...     pyplot.title('Akima interpolation of Gaussian noise')
...     pyplot.plot(x2, y2, 'r-', label='akima')
...     pyplot.plot(x2, y3, 'b:', label='scipy', linewidth=2.5)
...     pyplot.plot(x, y, 'go', label='data')
...     pyplot.legend()
...     pyplot.show()
>>> example()

"""

__version__ = '2020.1.1'

__all__ = ('interpolate',)

import numpy


def interpolate(x, y, x_new, axis=-1, out=None):
    """Return interpolated data using Akima's method.

    This Python implementation is inspired by the Matlab(r) code by
    N. Shamsundar. It lacks certain capabilities of the C implementation
    such as the output array argument and interpolation along an axis of a
    multidimensional data array.

    Parameters
    ----------
    x : array like
        1D array of monotonically increasing real values.
    y : array like
        N-D array of real values. y's length along the interpolation
        axis must be equal to the length of x.
    x_new : array like
        New independent variables.
    axis : int
        Specifies axis of y along which to interpolate. Interpolation
        defaults to last axis of y.
    out : array
        Optional array to receive results. Dimension at axis must equal
        length of x.

    Examples
    --------
    >>> interpolate([0, 1, 2], [0, 0, 1], [0.5, 1.5])
    array([-0.125,  0.375])
    >>> x = numpy.sort(numpy.random.random(10) * 10)
    >>> y = numpy.random.normal(0.0, 0.1, size=len(x))
    >>> z = interpolate(x, y, x)
    >>> numpy.allclose(y, z)
    True
    >>> x = x[:10]
    >>> y = numpy.reshape(y, (10, -1))
    >>> z = numpy.reshape(y, (10, -1))
    >>> interpolate(x, y, x, axis=0, out=z)
    >>> numpy.allclose(y, z)
    True

    """
    x = numpy.array(x, dtype='float64', copy=True)
    y = numpy.array(y, dtype='float64', copy=True)
    xi = numpy.array(x_new, dtype='float64', copy=True)

    if axis != -1 or out is not None or y.ndim != 1:
        raise NotImplementedError('implemented in C extension module')

    if x.ndim != 1 or xi.ndim != 1:
        raise ValueError('x-arrays must be one dimensional')

    n = len(x)
    if n < 3:
        raise ValueError('array too small')
    if n != y.shape[axis]:
        raise ValueError('size of x-array must match data shape')

    dx = numpy.diff(x)
    if any(dx <= 0.0):
        raise ValueError('x-axis not valid')

    if any(xi < x[0]) or any(xi > x[-1]):
        raise ValueError('interpolation x-axis out of bounds')

    m = numpy.diff(y) / dx
    mm = 2.0 * m[0] - m[1]
    mmm = 2.0 * mm - m[0]
    mp = 2.0 * m[n - 2] - m[n - 3]
    mpp = 2.0 * mp - m[n - 2]

    m1 = numpy.concatenate(([mmm], [mm], m, [mp], [mpp]))

    dm = numpy.abs(numpy.diff(m1))
    f1 = dm[2:n + 2]
    f2 = dm[0:n]
    f12 = f1 + f2

    ids = numpy.nonzero(f12 > 1e-9 * numpy.max(f12))[0]
    b = m1[1:n + 1]

    b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
    c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
    d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2

    bins = numpy.digitize(xi, x)
    bins = numpy.minimum(bins, n - 1) - 1
    bb = bins[0:len(xi)]
    wj = xi - x[bb]

    return ((wj * d[bb] + c[bb]) * wj + b[bb]) * wj + y[bb]


try:
    interpolate_py = interpolate
    from ._akima import interpolate
except ImportError:
    try:
        from _akima import interpolate
    except ImportError:
        import warnings

        warnings.warn('failed to import the _akima C extension module')
        del interpolate_py


if __name__ == '__main__':
    import doctest
    import random  # noqa: used in doctests

    doctest.testmod()
