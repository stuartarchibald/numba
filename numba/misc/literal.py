from numba.core.extending import overload
from numba.core import types
from numba.misc.special import literally, literal_unroll
from numba.core.errors import TypingError


@overload(literally)
def _ov_literally(obj):
    if isinstance(obj, types.Literal):
        # check of a __literal_repr__ first, the representation may not be the
        # same as the value, e.g. dict str:thing -> namedtuple
        literal_repr = getattr(obj, '__literal_repr__', None)
        if literal_repr is not None:
            lit = literal_repr
        else:
            lit = obj.literal_value
        return lambda obj: lit
    else:
        m = "Invalid use of non-Literal type in literally({})".format(obj)
        raise TypingError(m)


@overload(literal_unroll)
def literal_unroll_impl(container):

    def impl(container):
        return container
    return impl
