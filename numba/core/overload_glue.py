import numpy as np

class OverloadWrapper(object):

    def __init__(self, function=None):
        self._function = function
        self._BIND_TYPES = dict()
        self._selector = None
        self._TYPER = None
        # run to register overload, the intrinsic sorts out the binding to the
        # registered impls at the point the overload is evaluated, i.e. this
        # is all lazy.
        self._build()

    def wrap_typing(self, concrete_function):
        """
        Use this to replace @infer_global, it records the decorated function
        as a typer for the argument `concrete_function`.
        """
        assert concrete_function is self._function, "Wrong typing wrapper"
        # concrete_function is e.g. the numpy function this is implementing
        def inner(typing_class):
            # arg is the typing class
            self._TYPER = typing_class
            return typing_class
        return inner

    def wrap_impl(self, *args):
        """
        Use this to replace @lower*, it records the decorated function as the
        lowering implementation
        """
        # args is (concrete_function, *numba types)
        concrete_function = args[0]
        assert concrete_function is self._function, "Wrong impl wrapper"
        def inner(lowerer):
            self._BIND_TYPES[args[1:]] = lowerer
            return lowerer
        return inner

    def _assemble(self):
        """ Assembles the OverloadSelector definitions from the registered
        typing to lowering map.
        """
        from numba.core.base import OverloadSelector
        self._selector = OverloadSelector()
        msg = f"No entries in the typing->lowering map for {self._function}"
        assert self._BIND_TYPES, msg
        for sig, impl in self._BIND_TYPES.items():
            self._selector.append(impl, sig)

    def _build(self):
        from numba.core.extending import overload, intrinsic

        @overload(self._function)
        def ol_generated(*args):
            @intrinsic
            def intrin(tyctx, x, y):
                msg = f"No typer registered for {self._function}"
                assert self._TYPER is not None, msg
                typing = self._TYPER(tyctx)
                sig = typing.generic(args, {})
                if self._selector is None:
                    self._assemble()
                lowering = self._selector.find(sig.args)
                msg = (f"Could not find implementation to lower {sig} for ",
                       f"{self._function}")
                assert lowering is not None, msg
                return sig, lowering

            def jit_wrapper(*args):
                return intrin(*args)
            return jit_wrapper


class Gluer():
    def __init__(self):
        self._registered = dict()

    def __call__(self, func):
        if func in self._registered:
            return self._registered[func]
        else:
            wrapper = OverloadWrapper(func)
            self._registered[func] = wrapper
            return wrapper


overload_glue = Gluer()
del Gluer

