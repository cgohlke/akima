Akima Interpolation
===================

Akima is a Python library that implements Akima's interpolation method
using a continuously differentiable sub-spline built from piecewise cubic
polynomials [1]. The resultant curve passes through the given data points
and will appear smooth and natural.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:Version: 2019.1.1

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* `Matplotlib 2.2 <https://www.matplotlib.org>`_  (optional for plotting)

Notes
-----
The Akima module is no longer being actively developed.

Consider using `scipy.interpolate.Akima1DInterpolator
<http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ instead.

References
----------
(1) A new method of interpolation and smooth curve fitting based
    on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.

Examples
--------
>>> from scipy.interpolate import Akima1DInterpolator
>>> def example():
...     '''Plot interpolated Gaussian noise.'''
...     x = numpy.sort(numpy.random.random(10) * 100)
...     y = numpy.random.normal(0.0, 0.1, size=len(x))
...     x2 = numpy.arange(x[0], x[-1], 0.05)
...     y2 = interpolate(x, y, x2)
...     y3 = Akima1DInterpolator(x, y)(x2)
...     from matplotlib import pyplot
...     pyplot.title('Akima interpolation of Gaussian noise')
...     pyplot.plot(x2, y2, 'r-', label='akima')
...     pyplot.plot(x2, y3, 'b:', label='scipy', linewidth=2.5)
...     pyplot.plot(x, y, 'go', label='data')
...     pyplot.legend()
...     pyplot.show()
>>> example()
