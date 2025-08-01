..
  This file is generated by setup.py

Akima Interpolation
===================

Akima is a Python library that implements Akima's interpolation method
described in:

    A new method of interpolation and smooth curve fitting based on local
    procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.

A continuously differentiable sub-spline is built from piecewise cubic
polynomials. It passes through the given data points and will appear smooth
and natural.

This module is no longer being actively developed. Consider using
`scipy.interpolate.Akima1DInterpolator
<http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ instead.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2025.8.1

Quickstart
----------

Install the akima package and all dependencies from the
`Python Package Index <https://pypi.org/project/akima/>`_::

    python -m pip install -U akima

See `Examples`_ for using the programming interface.

Source code, examples, and support are available on
`GitHub <https://github.com/cgohlke/akima>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.11.9, 3.12.10, 3.13.5, 3.14.0rc 64-bit
- `NumPy <https://pypi.org/project/numpy/>`_ 2.3.2

Revisions
---------

2025.8.1

- Drop support for Python 3.10, support Python 3.14.

2025.1.1

- Drop support for Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.
- Support NumPy 2.

2024.1.6

- Add type hints.
- Drop support for Python 3.8 and numpy 1.22 (NEP 29).

2022.9.12

- Drop support for Python 3.7 (NEP 29).
- Update metadata.

Examples
--------
.. code-block:: python

    >>> import numpy
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
    ...
    >>> example()