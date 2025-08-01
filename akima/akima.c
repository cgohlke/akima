/* akima.c

/*
Copyright (c) 2007-2025, Christoph Gohlke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define _DOC_ \
"Interpolate data points in a plane based on Akima's method.\n\
\n\
Akima.c is a Python C extension module that provides a fast implementation\n\
for the akima package.\n\
\n\
Refer to the akima.py module for documentation and tests.\n\
\n\
:Author: `Christoph Gohlke <https://www.cgohlke.com/>`_\n\
:License: BSD-3-Clause\n\
:Version: 2025.8.1\n\
"

#define _VERSION_ "2025.8.1"

#define WIN32_LEAN_AND_MEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"


/*
A new method of interpolation and smooth curve fitting based on local
procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.
*/

int interpolate(
    Py_ssize_t si,            /* size of input arrays */
    char *xi, Py_ssize_t dxi, /* x coordinates and stride */
    char *yi, Py_ssize_t dyi, /* y coordinates and stride */
    Py_ssize_t so,            /* size of output arrays */
    char *xo, Py_ssize_t dxo, /* x coordinates of output and stride */
    char *yo, Py_ssize_t dyo, /* y output coordinates and stride */
    double *p  /* buffer for polynomial coefficients of size 4*si+4 */
    )
{
    Py_ssize_t i, s;
    double x0, x1, x2, x3;       /* extrapolated x values */
    double y0, y1, y2, y3;       /* extrapolated y values */
    double t0, t1, t2, t3;       /* temporary values */
    double d0, d1;
    double g0, g1;               /* gradients at extrapolated values */
    double *p0, *p1, *p2, *p3;   /* buffer pointers. p3 holds slopes */
    char *pxi, *pyi, *pxo, *pyo; /* data pointers */

    p0 = p;
    p1 = p + si + 1;
    p2 = p + si*2 + 2;
    p3 = p + si*3 + 3;

    /* slopes of input data */
    pxi = xi;
    pyi = yi;
    t0 = *((double *)pxi);
    t1 = *((double *)pyi);
    i = si-1;
    while (i--) {
        pxi += dxi;
        pyi += dyi;
        t0 = *((double *)pxi) - t0;
        if (t0 < 1e-12)
            return -1;
        *p3++ = (*((double *)pyi) - t1) / t0;
        t0 = *((double *)pxi);
        t1 = *((double *)pyi);
    }
    p3 = p + si*3 + 3;

    /* extrapolate 2 points on left side */
    t0 = *((double *)(xi));
    t1 = *((double *)(xi + dxi));
    t2 = *((double *)(yi));
    /* t3 = *((double *)(yi + dyi)); */

    x1 = t0 + t1 - *((double *)(xi+dxi+dxi));
    x0 = x1 + t0 - t1;
    y1 = (t0 - x1) * (p3[1] - 2.0*p3[0]) + t2;
    g1 = (t2 - y1) / (t0 - x1);
    y0 = (x1 - x0) * (p3[0] - 2.0*g1) + y1;
    g0 = (y1 - y0) / (x1 - x0);

    /* extrapolate 2 points on right side */
    s = (Py_ssize_t)xi + dxi*(si-1);
    t0 = *((double *)(s - dxi));
    t1 = *((double *)(s));
    x2 = t1 + t0 - *((double *)(s - dxi - dxi));
    x3 = x2 + t1 - t0;

    s = (Py_ssize_t)yi + dyi*(si-1);
    /* t2 = *((double *)(s - dyi)); */
    t3 = *((double *)(s));
    y2 = (2.0*p3[si-2] - p3[si-3]) * (x2 - t1) + t3;
    p3[si-1] = (y2 - t3) / (x2 - t1);
    y3 = (2.0*p3[si-1] - p3[si-2]) * (x3 - x2) + y2;
    p3[si] = (y3 - y2) / (x3 - x2);

    /* slopes */
    t1 = g0;
    t2 = g1;
    t3 = *p3;
    i = si;
    while (i--) {
        t0 = t1;
        t1 = t2;
        t2 = t3;
        t3 = *(p3 + 1);
        d0 = t3 - t2;
        if (d0 < 0.0)
            d0 *= -1.0;
        d1 = t1 - t0;
        if (d1 < 0.0)
            d1 *= -1.0;
        if ((d0 + d1) < 1e-9) {
            *p3++ = 0.5 * (t1 + t2);
        } else {
            *p3++ = (d0*t1 + d1*t2) / (d0 + d1);
        }
    }
    /* polynomial coefficients */
    pxi = xi;
    pyi = yi;
    t0 = *((double *)pxi);
    t1 = *((double *)pyi);
    p3 = p + si*3 + 3;
    g1 = *p3++;
    i = si;
    while (i--) {
        pxi += dxi;
        pyi += dyi;
        d0 = (*((double *)pxi) - t0);
        d1 = (*((double *)pyi) - t1);
        t2 = d1 / d0;
        g0 = g1;
        g1 = *p3++;
        *p0++ = t1;
        *p1++ = (3.0*t2 - 2.0*g0 - g1) / d0;
        *p2++ = (g0 + g1 - 2.0*t2) / (d0*d0);
        t0 = *((double *)pxi);
        t1 = *((double *)pyi);
    }
    /* interpolate data */
    p0 = p;
    p1 = p + si + 1;
    p2 = p + si*2 + 2;
    p3 = p + si*3 + 3;
    pxi = xi;
    pyi = yi;
    pxo = xo;
    pyo = yo;
    si -= 2;
    i = -1;
    s = so;
    while (s--) {
        t0 = *((double *)pxo);
        while ((t0 > *((double *)pxi)) && (i < si)) {
            pxi += dxi;
            i++;
        }
        if (i < 0) {
            i = 0;
            pxi = xi + dxi;
        }
        t1 = t0 - *((double *)(pxi - dxi));
        *((double *)pyo) = p0[i] + p3[i]*t1 + p1[i]*t1*t1 + p2[i]*t1*t1*t1;
        pyo += dyo;
        pxo += dxo;
    }
    return 0;
}

/*****************************************************************************/
/* Python functions */

/*
Numpy array converters for use with PyArg_Parse functions.
*/
static int
PyConverter_AnyDoubleArray(
    PyObject *object,
    PyObject **address)
{
    PyArrayObject *obj = (PyArrayObject *)object;
    if (PyArray_Check(object)
        && (PyArray_TYPE((const PyArrayObject *)object) == NPY_DOUBLE)) {
        *address = object;
        Py_INCREF(object);
        return NPY_SUCCEED;
    } else {
        *address = PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_ALIGNED);
        if (*address == NULL) {
            PyErr_Format(PyExc_ValueError, "can not convert to array");
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}

static int
PyOutputConverter_AnyDoubleArrayOrNone(
    PyObject *object,
    PyArrayObject **address)
{
    PyArrayObject *obj = (PyArrayObject *)object;
    if ((object == NULL) || (object == Py_None)) {
        *address = NULL;
        return NPY_SUCCEED;
    }
    if (PyArray_Check(object)
        && (PyArray_TYPE((const PyArrayObject *)object) == NPY_DOUBLE)) {
        Py_INCREF(object);
        *address = (PyArrayObject *)object;
        return NPY_SUCCEED;
    } else {
        PyErr_Format(PyExc_TypeError, "output must be array of type double");
        *address = NULL;
        return NPY_FAIL;
    }
}

/*
Interpolate array along axis using Akima's method.
*/
char py_interpolate_doc[] =
    "Return interpolated data along axis using Akima's method.";

static PyObject *
py_interpolate(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyArrayObject *xdata = NULL;
    PyArrayObject *data = NULL;
    PyArrayObject *xout = NULL;
    PyArrayObject *out = NULL;
    PyArrayObject *oout = NULL;
    PyArrayIterObject *dit = NULL;
    PyArrayIterObject *oit = NULL;
    npy_intp dstride, ostride, xdstride, xostride, size, outsize;
    Py_ssize_t newshape[NPY_MAXDIMS];
    int axis = NPY_MAXDIMS;
    int i, ndim, error;
    double *buffer = NULL;

    static char *kwlist[] = {"x", "y", "x_new", "axis", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&|O&O&", kwlist,
        PyConverter_AnyDoubleArray, &xdata,
        PyConverter_AnyDoubleArray, &data,
        PyConverter_AnyDoubleArray, &xout,
        PyArray_AxisConverter, &axis,
        PyOutputConverter_AnyDoubleArrayOrNone, &oout))
        goto _fail;

    /* check axis */
    ndim = PyArray_NDIM(data);
    if ((axis == NPY_MAXDIMS) || (axis == -1)) {
        axis = ndim - 1;
    } else if ((axis < 0) || (axis > NPY_MAXDIMS)) {
        PyErr_Format(PyExc_ValueError, "invalid axis");
        goto _fail;
    }

    if ((PyArray_NDIM(xdata) != 1) || (PyArray_NDIM(xout) != 1)) {
        PyErr_Format(PyExc_ValueError,
            "x-arrays must be one dimensional");
        goto _fail;
    }

    size = PyArray_DIM(data, axis);
    outsize = PyArray_DIM(xout, 0);

    if (size < 3) {
        PyErr_Format(PyExc_ValueError, "size along axis is too small");
        goto _fail;
    }

    if (size != PyArray_DIM(xdata, 0)) {
        PyErr_Format(PyExc_ValueError,
            "size of x-array must match data shape at axis");
        goto _fail;
    }

    for (i = 0; i < ndim; i++) {
        newshape[i] = (i == axis) ? outsize : PyArray_DIM(data, i);
    }

    if (oout == NULL) {
        /* create a new output array */
        out = (PyArrayObject*)PyArray_SimpleNew(ndim, newshape, NPY_DOUBLE);
        if (out == NULL) {
            PyErr_Format(PyExc_ValueError, "failed to allocate output array");
            goto _fail;
        }
    } else if (ndim != PyArray_NDIM(oout)) {
        PyErr_Format(PyExc_ValueError,
            "output and data array dimension mismatch");
        goto _fail;
    } else {
        for (i = 0; i < ndim; i++) {
            if (newshape[i] != PyArray_DIM(oout, i)) {
                PyErr_Format(PyExc_ValueError, "wrong output shape");
                goto _fail;
            }
        }
        out = oout;
    }

    /* iterate over all but specified axis */
    dit = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)data, &axis);
    oit = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    dstride = PyArray_STRIDE(data, axis);
    ostride = PyArray_STRIDE(out, axis);
    xdstride = PyArray_STRIDE(xdata, 0);
    xostride = PyArray_STRIDE(xout, 0);

    buffer = (double *)PyMem_Malloc((size * 4 + 4) * sizeof(double));
    if (buffer == NULL) {
        PyErr_Format(PyExc_ValueError, "failed to allocate output buffer");
        goto _fail;
    }

    while (dit->index < dit->size) {
        error = interpolate(
            size,
            PyArray_DATA(xdata), xdstride,
            dit->dataptr, dstride,
            outsize,
            PyArray_DATA(xout), xostride,
            oit->dataptr, ostride,
            buffer);

        if (error != 0) {
            PyErr_Format(PyExc_ValueError, "interpolate() failed");
            goto _fail;
        }

        PyArray_ITER_NEXT(oit);
        PyArray_ITER_NEXT(dit);
    }

    PyMem_Free(buffer);
    Py_DECREF(oit);
    Py_DECREF(dit);
    Py_DECREF(data);
    Py_DECREF(xout);
    Py_DECREF(xdata);

    /* Return output vector if not provided as argument */
    if (oout == NULL) {
        return PyArray_Return(out);
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }

  _fail:
    Py_XDECREF(xdata);
    Py_XDECREF(xout);
    Py_XDECREF(data);
    Py_XDECREF(oit);
    Py_XDECREF(dit);
    if (buffer != NULL)
        PyMem_Free(buffer);
    if (oout == NULL)
        Py_XDECREF(out);
    else
        Py_XDECREF(oout);
    return NULL;
}


/*****************************************************************************/
/* Python module */

static PyMethodDef module_methods[] = {
    {"interpolate", (PyCFunction)py_interpolate, METH_VARARGS|METH_KEYWORDS,
        py_interpolate_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};


struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_akima",
    NULL,
    sizeof(struct module_state),
    module_methods,
    NULL,
    module_traverse,
    module_clear,
    NULL
};

PyMODINIT_FUNC
PyInit__akima(void)
{
    PyObject *module;

    char *doc = (char *)PyMem_Malloc(sizeof(_DOC_) + sizeof(_VERSION_));
    PyOS_snprintf(doc, sizeof(_DOC_) + sizeof(_VERSION_), _DOC_, _VERSION_);

    moduledef.m_doc = doc;
    module = PyModule_Create(&moduledef);

    PyMem_Free(doc);

    if (module == NULL) {
        return NULL;
    }

#ifdef Py_GIL_DISABLED
    /* this module supports running with the GIL disabled */
    if (PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED) < 0) {
        Py_DECREF(module);
        return NULL;
    }
#endif

    if (_import_array() < 0) {
        Py_DECREF(module);
        return NULL;
    }

    {
    PyObject *s = PyUnicode_FromString(_VERSION_);
    PyObject *dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);
    }

    return module;
}
