from numba.tests.support import TestCase

class TestAPIMoves_Q1_2020(TestCase):
    """ Checks API moves made in Q1 2020, this roughly corresponds to 0.48->0.49
    """
    def test_numba_utils(self):
        from numba import utils
        from numba.utils import pysignature
        from numba.utils import OPERATORS_TO_BUILTINS

    def test_numba_untyped_passes(self):
        import numba.untyped_passes
        from numba.untyped_passes import InlineClosureLikes

    def test_numba_unsafe(self):
        import numba.unsafe
        from numba.unsafe import refcount
        from numba.unsafe.ndarray import empty_inferred

    def test_numba_unicode(self):
        import numba.unicode
        from numba.unicode import unbox_unicode_str
        from numba.unicode import make_string_from_constant
        from numba.unicode import _slice_span
        from numba.unicode import _normalize_slice
        from numba.unicode import _empty_string
        from numba.unicode import PY_UNICODE_1BYTE_KIND
        from numba.unicode import PY_UNICODE_2BYTE_KIND
        from numba.unicode import PY_UNICODE_4BYTE_KIND
        from numba.unicode import PY_UNICODE_WCHAR_KIND

    def test_numba_typing(self):
        import numba.typing.typeof
        from numba.typing.typeof import typeof_impl
        from numba.typing.typeof import _typeof_ndarray
        import numba.typing.templates
        from numba.typing.templates import signature
        from numba.typing.templates import infer_global
        from numba.typing.templates import infer_getattr
        from numba.typing.templates import infer
        from numba.typing.templates import bound_function
        from numba.typing.templates import AttributeTemplate
        from numba.typing.templates import AbstractTemplate
        import numba.typing.npydecl
        from numba.typing.npydecl import supported_ufuncs
        from numba.typing.npydecl import NumpyRulesInplaceArrayOperator
        from numba.typing.npydecl import NumpyRulesArrayOperator
        from numba.typing.npydecl import NdConcatenate
        from numba import typing
        from numba.typing import fold_arguments
        from numba.typing import ctypes_utils
        from numba.typing.ctypes_utils import make_function_type
        from numba.typing.collections import GetItemSequence
        from numba.typing import builtins
        from numba.typing.builtins import MinMaxBase
        from numba.typing.builtins import IndexValueType
        from numba.typing import arraydecl
        from numba.typing.arraydecl import get_array_index_type
        from numba.typing.arraydecl import ArrayAttribute
        ArrayAttribute.resolve_var
        ArrayAttribute.resolve_sum
        ArrayAttribute.resolve_prod
        from numba.typing import Signature
        numba.typing.templates.infer_global

    def test_numba_typeinfer(self):
        import numba.typeinfer
        from numba.typeinfer import IntrinsicCallConstraint

    def test_numba_typedobjectutils(self):
        import numba.typedobjectutils

    def test_numba_typed_passes(self):
        import numba.typed_passes
        from numba.typed_passes import type_inference_stage
        from numba.typed_passes import AnnotateTypes

    def test_numba_typed(self):
        # no refactoring was done to `numba.typed`.
        import numba.typed
        from numba.typed import List
        List.empty_list
        from numba.typed import Dict
        Dict.empty

    def test_numba_typeconv(self):
        from numba import typeconv

    def test_numba_targets(self):
        import numba.targets
        from numba.targets import ufunc_db
        from numba.targets.ufunc_db import get_ufuncs
        import numba.targets.slicing
        from numba.targets.slicing import guard_invalid_slice
        from numba.targets.slicing import get_slice_length
        from numba.targets.slicing import fix_slice
        from numba.targets import setobj
        from numba.targets.setobj import set_empty_constructor
        from numba.targets import registry
        from numba.targets.registry import dispatcher_registry
        from numba.targets.registry import cpu_target
        typing_context = cpu_target.typing_context
        typing_context.resolve_value_type
        from numba.targets import options
        from numba.targets.options import TargetOptions
        from numba.targets import npdatetime
        import numba.targets.listobj
        from numba.targets.listobj import ListInstance
        ListInstance.allocate
        import numba.targets.imputils
        from numba.targets.imputils import lower_cast
        from numba.targets.imputils import iternext_impl
        from numba.targets.imputils import impl_ret_new_ref
        from numba.targets.imputils import RefType
        import numba.targets.hashing
        import numba.targets.cpu
        from numba.targets.cpu import ParallelOptions
        from numba.targets.cpu import CPUTargetOptions
        from numba.targets.cpu import CPUContext
        CPUTargetOptions.OPTIONS
        import numba.targets.callconv
        import numba.targets.builtins
        from numba.targets.builtins import get_type_min_value
        from numba.targets.builtins import get_type_max_value
        import numba.targets.boxing
        import numba.targets.arrayobj
        from numba.targets.arrayobj import store_item
        from numba.targets.arrayobj import setitem_array
        from numba.targets.arrayobj import populate_array
        from numba.targets.arrayobj import numpy_empty_nd
        from numba.targets.arrayobj import make_array
        from numba.targets.arrayobj import getiter_array
        from numba.targets.arrayobj import getitem_arraynd_intp
        from numba.targets.arrayobj import getitem_array_tuple
        from numba.targets.arrayobj import fancy_getitem_array
        from numba.targets.arrayobj import array_reshape
        from numba.targets.arrayobj import array_ravel
        from numba.targets.arrayobj import array_len
        from numba.targets.arrayobj import array_flatten
        import numba.targets.arraymath
        from numba.targets.arraymath import get_isnan

    def test_numba_stencilparfor(self):
        from numba.stencilparfor import _compute_last_ind

    def test_numba_stencil(self):
        from numba import stencil
        from numba.stencil import StencilFunc

    def test_numba_runtime(self):
        import numba.runtime
        import numba.runtime.nrt
        from numba.runtime import nrt

    def test_numba_rewrites(self):
        from numba import rewrites
        from numba.rewrites import rewrite_registry
        rewrite_registry.apply

    def test_numba_rewrites(self):
        from numba import pythonapi
        from numba.pythonapi import _UnboxContext
        from numba.pythonapi import _BoxContext

    def test_numba_parfor(self):
        from numba import parfor
        import numba.parfor
        from numba.parfor import replace_functions_map
        from numba.parfor import min_checker
        from numba.parfor import maximize_fusion
        from numba.parfor import max_checker
        from numba.parfor import lower_parfor_sequential
        from numba.parfor import internal_prange
        from numba.parfor import init_prange
        from numba.parfor import argmin_checker
        from numba.parfor import argmax_checker
        from numba.parfor import PreParforPass
        from numba.parfor import ParforPass
        from numba.parfor import Parfor

    def test_numba_numpy_support(self):
        import numba.numpy_support
        from numba import numpy_support
        from numba.numpy_support import map_layout
        from numba.numpy_support import from_dtype
        from numba.numpy_support import as_dtype

    def test_numba_npdatetime(self):
        import numba.npdatetime

    def test_numba_lowering(self):
        import numba.lowering

    def test_numba_ir_utils(self):
        import numba.ir_utils
        from numba.ir_utils import simplify_CFG
        from numba.ir_utils import replace_vars_stmt
        from numba.ir_utils import replace_arg_nodes
        from numba.ir_utils import remove_dead
        from numba.ir_utils import remove_call_handlers
        from numba.ir_utils import next_label
        from numba.ir_utils import mk_unique_var
        from numba.ir_utils import get_ir_of_code
        from numba.ir_utils import get_definition
        from numba.ir_utils import find_const
        from numba.ir_utils import dprint_func_ir
        from numba.ir_utils import compute_cfg_from_blocks
        from numba.ir_utils import compile_to_numba_ir
        from numba.ir_utils import build_definitions
        from numba.ir_utils import alias_func_extensions
        from numba.ir_utils import _max_label
        from numba.ir_utils import _add_alias
        from numba.ir_utils import GuardException

    def test_numba_ir(self):
        import numba.ir
        from numba import ir
        from numba.ir import Assign, Const, Expr, Var

    def test_numba_inline_closurecall(self):
        import numba.inline_closurecall
        from numba import inline_closurecall
        from numba.inline_closurecall import inline_closure_call
        from numba.inline_closurecall import _replace_returns
        from numba.inline_closurecall import _replace_freevars
        from numba.inline_closurecall import _add_definitions

    def test_numba_errors(self):
        import numba.errors
        from numba.errors import WarningsFixer
        from numba.errors import TypingError
        from numba.errors import NumbaWarning
        from numba.errors import ForceLiteralArg

    def test_numba_dispatcher(self):
        import numba.dispatcher
        from numba.dispatcher import ObjModeLiftedWith
        from numba.dispatcher import Dispatcher

    def test_numba_dictobject(self):
        from numba.dictobject import DictModel

    def test_numba_datamodel(self):
        from numba import datamodel
        import numba.datamodel
        from numba.datamodel import registry
        registry.register_default
        from numba.datamodel import register_default
        from numba.datamodel import models
        from numba.datamodel.models import StructModel
        from numba.datamodel.models import PointerModel
        from numba.datamodel.models import BooleanModel

    def test_numba_compiler_machinery(self):
        import numba.compiler_machinery

    def test_numba_compiler(self):
        import numba.compiler
        from numba.compiler import run_frontend
        from numba.compiler import StateDict
        from numba.compiler import Flags
        Flags.OPTIONS
        from numba.compiler import DefaultPassBuilder
        DefaultPassBuilder.define_nopython_pipeline
        from numba.compiler import CompilerBase

    def test_numba_cgutils(self):
        import numba.cgutils
        from numba.cgutils import unpack_tuple
        from numba.cgutils import true_bit
        from numba.cgutils import is_not_null
        from numba.cgutils import increment_index
        from numba.cgutils import get_null_value
        from numba.cgutils import false_bit
        from numba.cgutils import create_struct_proxy
        from numba.cgutils import alloca_once_value
        from numba.cgutils import alloca_once

    def test_numba_array_analysis(self):
        import numba.array_analysis
        from numba.array_analysis import array_analysis_extensions
        from numba.array_analysis import ArrayAnalysis
        args = ((None,) * 4)
        ArrayAnalysis(*args)._analyze_op_static_getitem
        ArrayAnalysis(*args)._analyze_broadcast

    def test_numba_analysis(self):
        import numba.analysis
        from numba.analysis import ir_extension_usedefs
        from numba.analysis import compute_use_defs
        from numba.analysis import compute_cfg_from_blocks
        from numba.analysis import _use_defs_result

    def test_numba_jitclass(self):
        from numba import jitclass
        @jitclass
        class foo():
            pass

if __name__ == '__main__':
    unittest.main()
