Akima Interpolation
===================

Akima is a Python library that implements Akima's interpolation method
described in:

    A new method of interpolation and smooth curve fitting based
    on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.

A continuously differentiable sub-spline is built from piecewise cubic
polynomials. It passes through the given data points and will appear smooth
and natural.

This module is no longer being actively developed. Consider using
`scipy.interpolate.Akima1DInterpolator
<http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ instead.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2022.9.12

Requirements
------------

This release has been tested with the following requirements and dependencies
(other versions may work):

- `CPython 3.8.10, 3.9.13, 3.10.7, 3.11.0rc2 <https://www.python.org>`_
- `NumPy 1.22.4 <https://pypi.org/project/numpy/>`_

Revisions
---------

2022.9.12

- Remove support for Python 3.7 (NEP 29).
- Update metadata.

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
