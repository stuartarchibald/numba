.. _cfunc:

====================================
Creating C callbacks with ``@cfunc``
====================================

Interfacing with some native libraries (for example written in C or C++)
can necessitate writing native callbacks to provide business logic to the
library.  The :func:`numba.cfunc` decorator creates a compiled function
callable from foreign C code, using the signature of your choice.


Basic usage
===========

The ``@cfunc`` decorator has a similar usage to ``@jit``, but with an
important difference: passing a single signature is mandatory.
It determines the visible signature of the C callback::

   from numba import cfunc

   @cfunc("float64(float64, float64)")
   def add(x, y):
       return x + y


The C function object exposes the address of the compiled C callback as
the :attr:`~CFunc.address` attribute, so that you can pass it to any
foreign C or C++ library (see C library example below).  It also exposes
a :mod:`ctypes` callback object pointing to that callback; that object is
also callable from Python, making it easy to check the compiled code::

   @cfunc("float64(float64, float64)")
   def add(x, y):
       return x + y

   print(add.ctypes(4.0, 5.0))  # prints "9.0"


Example
=======

In this example, we are going to be using the ``scipy.integrate.quad``
function.  That function accepts either a regular Python callback or
a C callback wrapped in a :mod:`ctypes` callback object.

Let's define a pure Python integrand and compile it as a
C callback::

   >>> import numpy as np
   >>> from numba import cfunc
   >>> def integrand(t):
           return np.exp(-t) / t**2
      ...:
   >>> nb_integrand = cfunc("float64(float64)")(integrand)

We can pass the ``nb_integrand`` object's :mod:`ctypes` callback to
``scipy.integrate.quad`` and check that the results are the same as with
the pure Python function::

   >>> import scipy.integrate as si
   >>> def do_integrate(func):
           """
           Integrate the given function from 1.0 to +inf.
           """
           return si.quad(func, 1, np.inf)
      ...:
   >>> do_integrate(integrand)
   (0.14849550677592208, 3.8736750296130505e-10)
   >>> do_integrate(nb_integrand.ctypes)
   (0.14849550677592208, 3.8736750296130505e-10)


Using the compiled callback, the integration function does not invoke the
Python interpreter each time it evaluates the integrand.  In our case, the
integration is made 18 times faster::

   >>> %timeit do_integrate(integrand)
   1000 loops, best of 3: 242 µs per loop
   >>> %timeit do_integrate(nb_integrand.ctypes)
   100000 loops, best of 3: 13.5 µs per loop


Dealing with pointers and array memory
======================================

A less trivial use case of C callbacks involves doing operation on some
array of data passed by the caller.  As C doesn't have a high-level
abstraction similar to Numpy arrays, the C callback's signature will pass
low-level pointer and size arguments.  Nevertheless, the Python code for
the callback will expect to exploit the power and expressiveness of Numpy
arrays.

In the following example, the C callback is expected to operate on 2-d arrays,
with the signature ``void(double *input, double *output, int m, int n)``.
You can implement such a callback thusly::

   from numba import cfunc, types, carray

   c_sig = types.void(types.CPointer(types.double),
                      types.CPointer(types.double),
                      types.intc, types.intc)

   @cfunc(c_sig)
   def my_callback(in_, out, m, n):
       in_array = carray(in_, (m, n))
       out_array = carray(out, (m, n))
       for i in range(m):
           for j in range(n):
               out_array[i, j] = 2 * in_array[i, j]


The :func:`numba.carray` function takes as input a data pointer and a shape
and returns an array view of the given shape over that data.  The data is
assumed to be laid out in C order.  If the data is laid out in Fortran order,
:func:`numba.farray` should be used instead.


Signature specification
=======================

The explicit ``@cfunc`` signature can use any :ref:`Numba types <numba-types>`,
but only a subset of them make sense for a C callback.  You should
generally limit yourself to scalar types (such as ``int8`` or ``float64``)
or pointers to them (for example ``types.CPointer(types.int8)``).


Compilation options
===================

A number of keyword-only arguments can be passed to the ``@cfunc``
decorator: ``nopython`` and ``cache``.  Their meaning is similar to those
in the ``@jit`` decorator.

C Library Example
=================

As noted above the address of the callback is exposed via the 
:attr:`~CFunc.address` attribute. As a result it is possible to use this
to call the function directly from C/C++ code. This example creates two
``cfunc`` functions, both with the same signature, and then calls them from a C
extension library (Python 2.7 style API):

.. code-block:: c

    #include "Python.h"
    #include <stdio.h>

    // signature to match the Python cfunc signature
    typedef double (*func_t)(double, double);

    static PyObject * call_cfunc(PyObject *self, PyObject *args) {
        PyObject *obj = NULL, *ret = NULL;
        func_t func_inst;
        double res;
        
        // unpack the argument
        if(!PyArg_ParseTuple(args, "O", &obj)) return NULL;
        
        // get the address and assign it to the function pointer
        func_inst = PyLong_AsVoidPtr(obj);
        if(PyErr_Occurred()) return NULL;
        
        // make the call
        res = func_inst(10.0, 20.0);
        printf("C: JITed func result: %f\n",res);

        // return the result
        ret = PyFloat_FromDouble(res);
        if(PyErr_Occurred()) return NULL;
        return ret;
    }

    static PyMethodDef methods[] = {
        {"call_cfunc", call_cfunc, METH_VARARGS,
        "Execute JITed function"},
        {NULL, NULL, 0, NULL}        // sentinel
    };

    PyMODINIT_FUNC
    initjit_cfuncs(void)
    {
        PyObject *m = NULL;
        m = Py_InitModule("jit_cfuncs", methods);
        if(!m) {printf("C: Could not initialize module!\n");}
    }

Compile and link with something like:

.. code-block:: bash

    $ gcc jit_cfuncs.c -fPIC -c -o jit_cfuncs.o 
    $ gcc -shared -fPIC jit_cfuncs.o -o jit_cfuncs.so -lpython2.7

Finally, create a Python function::

    from numba import cfunc

    # define some functions to cfunc
    def add(a, b):
        return a + b

    def sub(a, b):
        return a - b

    # Create cfunc callbacks
    sig = "float64(float64, float64)"
    add_nb = cfunc(sig)(add)
    sub_nb = cfunc(sig)(sub)

    # demonstrate the callback via the ctypes object
    print("Python: ctypes: %s" % add_nb.ctypes(5, 10))
    print("Python: ctypes: %s" % sub_nb.ctypes(5, 10))

    # import the extension module
    import jit_cfuncs

    # pass the cfunc address for use in a C function pointer
    print("Python: cfunc result: %s" % jit_cfuncs.call_cfunc(add_nb.address))
    print("Python: cfunc result: %s" % jit_cfuncs.call_cfunc(sub_nb.address))
    
Executing gives:

.. code-block:: bash

    $ python jit_cfuncs.py 
    Python: ctypes: 15.0
    Python: ctypes: -5.0
    C: cfunc result: 30.000000
    Python: cfunc result: 30.0
    C: cfunc result: -10.000000
    Python: cfunc result: -10.0




