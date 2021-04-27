import cmath

from numba.core import types, utils
from numba.core.typing.templates import (AbstractTemplate, ConcreteTemplate,
                                    signature, Registry)

from numba.core.overload_glue import glue_typing

# TODO: support non-complex arguments (floats and ints)


@glue_typing(cmath.acos)
@glue_typing(cmath.acosh)
@glue_typing(cmath.asin)
@glue_typing(cmath.asinh)
@glue_typing(cmath.atan)
@glue_typing(cmath.atanh)
@glue_typing(cmath.cos)
@glue_typing(cmath.cosh)
@glue_typing(cmath.exp)
@glue_typing(cmath.log10)
@glue_typing(cmath.sin)
@glue_typing(cmath.sinh)
@glue_typing(cmath.sqrt)
@glue_typing(cmath.tan)
@glue_typing(cmath.tanh)
class CMath_unary(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in sorted(types.complex_domain)]


@glue_typing(cmath.isinf)
@glue_typing(cmath.isnan)
class CMath_predicate(ConcreteTemplate):
    cases = [signature(types.boolean, tp) for tp in
             sorted(types.complex_domain)]


@glue_typing(cmath.isfinite)
class CMath_isfinite(CMath_predicate):
    pass


@glue_typing(cmath.log)
class Cmath_log(ConcreteTemplate):
    # unary cmath.log()
    cases = [signature(tp, tp) for tp in sorted(types.complex_domain)]
    # binary cmath.log()
    cases += [signature(tp, tp, tp) for tp in sorted(types.complex_domain)]


@glue_typing(cmath.phase)
class Cmath_phase(ConcreteTemplate):
    cases = [signature(tp, types.complex128) for tp in [types.float64]]
    cases += [signature(types.float32, types.complex64)]


@glue_typing(cmath.polar)
class Cmath_polar(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [tp] = args
        if tp in types.complex_domain:
            float_type = tp.underlying_float
            return signature(types.UniTuple(float_type, 2), tp)


@glue_typing(cmath.rect)
class Cmath_rect(ConcreteTemplate):
    cases = [signature(types.complex128, tp, tp)
             for tp in [types.float64]]
    cases += [signature(types.complex64, types.float32, types.float32)]
