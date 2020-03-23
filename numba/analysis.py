from numba.core.errors import (deprecate_moved_module,
                               deprecate_moved_module_getattr)
from numba.core.analysis import * # noqa: F403, F401
from numba.core.analysis import _use_defs_result # noqa: F401

deprecate_moved_module(__name__, 'numba.core.analysis')
__getattr__ = deprecate_moved_module_getattr(__name__, 'numba.core.analysis')
