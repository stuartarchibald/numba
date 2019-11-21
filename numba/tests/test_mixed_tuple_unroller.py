from __future__ import print_function, absolute_import, division

import numpy as np

from numba.tests.support import TestCase, MemoryLeakMixin
from numba import njit, types, typed, ir, errors
from numba.testing import unittest
from numba.extending import overload
from numba.compiler_machinery import PassManager, register_pass, FunctionPass
from numba.compiler import CompilerBase
from numba.untyped_passes import (TranslateByteCode, FixupArgs, IRProcessing,
                                  SimplifyCFG, IterLoopCanonicalization)
from numba.typed_passes import (NopythonTypeInference, IRLegalization,
                                NoPythonBackend, PartialTypeInference)
from numba.ir_utils import (compute_cfg_from_blocks, find_topo_order,
                            flatten_labels)


class TestLiteralTupleInterpretation(MemoryLeakMixin, TestCase):

    def check(self, func, var):
        cres = func.overloads[func.signatures[0]]
        ty = cres.fndesc.typemap[var]
        self.assertTrue(isinstance(ty, types.Tuple))
        for subty in ty:
            self.assertTrue(isinstance(subty, types.Literal), "non literal")

    def test_homogeneous_literal(self):
        @njit
        def foo():
            x = (1, 2, 3)
            return x[1]

        self.assertEqual(foo(), foo.py_func())
        self.check(foo, 'x')

    def test_heterogeneous_literal(self):
        @njit
        def foo():
            x = (1, 2, 3, 'a')
            return x[3]

        self.assertEqual(foo(), foo.py_func())
        self.check(foo, 'x')

    def test_non_literal(self):
        @njit
        def foo():
            x = (1, 2, 3, 'a', 1j)
            return x[4]

        self.assertEqual(foo(), foo.py_func())
        with self.assertRaises(AssertionError) as e:
            self.check(foo, 'x')

        self.assertIn("non literal", str(e.exception))


@register_pass(mutates_CFG=False, analysis_only=True)
class PreserveIR(FunctionPass):
    _name = "preserve_ir"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.metadata['func_ir'] = state.func_ir
        return False


@register_pass(mutates_CFG=False, analysis_only=False)
class ResetTypeInfo(FunctionPass):
    _name = "reset_the_type_information"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.typemap = None
        state.return_type = None
        state.calltypes = None
        return True


class TestLoopCanonicalisation(MemoryLeakMixin, TestCase):

    def get_pipeline(use_canonicaliser, use_partial_typing=False):
        class NewCompiler(CompilerBase):

            def define_pipelines(self):
                pm = PassManager("custom_pipeline")

                # untyped
                pm.add_pass(TranslateByteCode, "analyzing bytecode")
                pm.add_pass(IRProcessing, "processing IR")
                if use_partial_typing:
                    pm.add_pass(PartialTypeInference, "do partial typing")
                if use_canonicaliser:
                    pm.add_pass(IterLoopCanonicalization, "Canonicalise loops")
                pm.add_pass(SimplifyCFG, "Simplify the CFG")

                # typed
                if use_partial_typing:
                    pm.add_pass(ResetTypeInfo, "resets the type info state")

                pm.add_pass(NopythonTypeInference, "nopython frontend")

                # legalise
                pm.add_pass(IRLegalization, "ensure IR is legal")

                # preserve
                pm.add_pass(PreserveIR, "save IR for later inspection")

                # lower
                pm.add_pass(NoPythonBackend, "nopython mode backend")

                # finalise the contents
                pm.finalize()

                return [pm]
        return NewCompiler

    # generate variants
    LoopIgnoringCompiler = get_pipeline(False)
    LoopCanonicalisingCompiler = get_pipeline(True)
    TypedLoopCanonicalisingCompiler = get_pipeline(True, True)

    def test_simple_loop_in_depth(self):
        """ This heavily checks a simple loop transform """

        def get_info(pipeline):
            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for i in tup:
                    acc += i
                return acc

            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['func_ir']
            return func_ir, cres.fndesc

        ignore_loops_ir, ignore_loops_fndesc = \
            get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = \
            get_info(self.LoopCanonicalisingCompiler)

        # check CFG is the same
        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)

        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)

        # check there's three more call types in the canonicalised one:
        # len(tuple arg)
        # range(of the len() above)
        # getitem(tuple arg, index)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3,
                         len(canonicalise_loops_fndesc.calltypes))

        def find_getX(fd, op):
            return [x for x in fd.calltypes.keys() 
                    if isinstance(x, ir.Expr) and x.op == op]

        il_getiters = find_getX(ignore_loops_fndesc, "getiter")
        self.assertEqual(len(il_getiters), 1) # tuple iterator

        cl_getiters = find_getX(canonicalise_loops_fndesc, "getiter")
        self.assertEqual(len(cl_getiters), 1) # loop range iterator

        cl_getitems = find_getX(canonicalise_loops_fndesc, "getitem")
        self.assertEqual(len(cl_getitems), 1) # tuple getitem induced by loop

        # check the value of the untransformed IR getiter is now the value of
        # the transformed getitem
        self.assertEqual(il_getiters[0].value.name, cl_getitems[0].value.name)

        # check the type of the transformed IR getiter is a range iter
        range_inst = canonicalise_loops_fndesc.calltypes[cl_getiters[0]].args[0]
        self.assertTrue(isinstance(range_inst, types.RangeType))

    def test_influence_of_transform(self):
        """ This heavily checks a a typed transformation only impacts tuple
        induced loops"""

        def get_info(pipeline):
            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for y in (1, 2, 3):
                    acc += 1
                return acc

            import dis
            print(dis.dis(foo.py_func))

            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['func_ir']
            return func_ir, cres.fndesc

        ignore_loops_ir, ignore_loops_fndesc = \
            get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = \
            get_info(self.LoopCanonicalisingCompiler)

        # check CFG is the same
        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)

        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)

        # check there's three more call types in the canonicalised one:
        # len(tuple arg)
        # range(of the len() above)
        # getitem(tuple arg, index)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3,
                         len(canonicalise_loops_fndesc.calltypes))

        def find_getX(fd, op):
            return [x for x in fd.calltypes.keys() 
                    if isinstance(x, ir.Expr) and x.op == op]

        il_getiters = find_getX(ignore_loops_fndesc, "getiter")
        self.assertEqual(len(il_getiters), 1) # tuple iterator

        cl_getiters = find_getX(canonicalise_loops_fndesc, "getiter")
        self.assertEqual(len(cl_getiters), 1) # loop range iterator

        cl_getitems = find_getX(canonicalise_loops_fndesc, "getitem")
        self.assertEqual(len(cl_getitems), 1) # tuple getitem induced by loop

        # check the value of the untransformed IR getiter is now the value of
        # the transformed getitem
        self.assertEqual(il_getiters[0].value.name, cl_getitems[0].value.name)

        # check the type of the transformed IR getiter is a range iter
        range_inst = canonicalise_loops_fndesc.calltypes[cl_getiters[0]].args[0]
        self.assertTrue(isinstance(range_inst, types.RangeType))


    def test_transform_will_do_illegal_things(self):
        """ This checks the transform, when there's no typemap, will happily
        transform a range induced loop into something illegal/can't be compiled
        """
        with self.assertRaises(errors.TypingError) as e:
            @njit(pipeline_class=self.LoopCanonicalisingCompiler)
            def foo():
                for _ in range(10):
                    pass
            foo()

        self.assertIn("Invalid use of", str(e.exception))
        self.assertIn("range_state_int", str(e.exception))

    def test_lots_of_loops(self):
        """ This heavily checks a simple loop transform """

        def get_info(pipeline):
            @njit(pipeline_class=pipeline)
            def foo(tup):
                acc = 0
                for i in tup:
                    acc += i
                    for j in tup:
                        acc += 1
                        if j > 5:
                            break
                    else:
                        acc -= 2
                for i in tup:
                    acc -= i

                return acc

            x = (1, 2, 3)
            self.assertEqual(foo(x), foo.py_func(x))
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['func_ir']
            return func_ir, cres.fndesc

        ignore_loops_ir, ignore_loops_fndesc = \
            get_info(self.LoopIgnoringCompiler)
        canonicalise_loops_ir, canonicalise_loops_fndesc = \
            get_info(self.LoopCanonicalisingCompiler)

        # check CFG is the same
        def compare_cfg(a, b):
            a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
            b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
            self.assertEqual(a_cfg, b_cfg)

        compare_cfg(ignore_loops_ir, canonicalise_loops_ir)

        # check there's three * N more call types in the canonicalised one:
        # len(tuple arg)
        # range(of the len() above)
        # getitem(tuple arg, index)
        self.assertEqual(len(ignore_loops_fndesc.calltypes) + 3 * 3,
                         len(canonicalise_loops_fndesc.calltypes))



class TestMixedTupleUnroll(MemoryLeakMixin, TestCase):
    def test_1(self):

        @njit
        def foo(idx, z):
            a = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for i in range(len(a)):
                acc += a[i]
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc

        f = 9
        k = f

        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_2(self):

        @njit
        def foo(idx, z):
            x = (12, 12.7, 3j, 4, z, 2 * z)
            acc = 0
            for a in x:
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    break
            return acc

        f = 9
        k = f

        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_3(self):

        @njit
        def foo(idx, z):
            x = (12, 12.7, 3j, 4, z, 2 * z)
            y = ('foo', z, 2 * z)
            acc = 0
            for a in x:
                acc += a
                if acc.real < 26:
                    acc -= 1
                else:
                    for t in y:
                        acc += t is False
                    break
            return acc

        f = 9
        k = f

        self.assertEqual(foo(2, k), foo.py_func(2, k))

    def test_4(self):

        @njit
        def foo(tup):
            acc = 0
            for a in tup:
                acc += a.sum()
            return acc

        n = 10
        tup = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)))
        self.assertEqual(foo(tup), foo.py_func(tup))

    def test_5(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            for a in tup1:
                if a == 'a':
                    acc += tup2[0].sum()
                elif a == 'b':
                    acc += tup2[1].sum()
                elif a == 'c':
                    acc += tup2[2].sum()
                elif a == 12:
                    acc += tup2[3].sum()
                elif a == 3j:
                    acc += tup2[3].sum()
                elif a == ('f',):
                    acc += tup2[3].sum()
                else:
                    raise RuntimeError("Unreachable")
            return acc

        n = 10
        tup1 = ('a', 'b', 'c', 12, 3j, ('f',))
        tup2 = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)),
                np.ones((n, n, n, n)), np.ones((n, n, n, n, n)))
        self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))

    @unittest.skip("needs more clever branch prune")
    def test_6(self):
        # This wont work because both sides of the branch need typing as neither
        # can be pruned by the current pruner
        @njit
        def foo(tup):
            acc = 0
            idx = 0
            str_buf = []
            for a in tup:
                if a == 'a':
                    str_buf.append(a)
                else:
                    acc += a
            return acc

        tup = ('a', 12)
        self.assertEqual(foo(tup), foo.py_func(tup))

    def test_7(self):

        @njit
        def foo(tup):
            acc = 0
            for a in tup:
                acc += len(a)
            return acc

        n = 10
        tup = (np.ones((n,)), np.ones((n, n)), "ABCDEFGHJI", (1, 2, 3),
               (1, 'foo', 2, 'bar'), {3})
        self.assertEqual(foo(tup), foo.py_func(tup))

    def test_8(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            for a in tup1:
                if a == 'a':
                    acc += tup2[0]()
                elif a == 'b':
                    acc += tup2[1]()
                elif a == 'c':
                    acc += tup2[2]()
            return acc

        def gen(x):
            def impl():
                return x
            return njit(impl)

        n = 10
        tup1 = ('a', 'b', 'c', 12, 3j, ('f',))
        tup2 = (gen(1), gen(2), gen(3))
        self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))

    @unittest.skip("Fail. Is it possible to have index mixed tuple on the RHS?")
    def test_9(self):

        @njit
        def foo(tup1, tup2):
            acc = 0
            idx = 0
            for a in tup1:
                if a == 'a':
                    acc += tup2[idx]
                elif a == 'b':
                    acc += tup2[idx]
                elif a == 'c':
                    acc += tup2[idx]
                idx += 1
            return idx, acc

        @njit
        def func1():
            return 1

        @njit
        def func2():
            return 2

        @njit
        def func3():
            return 3

        n = 10
        tup1 = ('a', 'b', 'c')
        #tup2 = (func1, func2, func3)
        tup2 = (1j, 1, 2)
        self.assertEqual(foo(tup1, tup2), foo.py_func(tup1, tup2))

    def test_10(self):

        def dt(value):
            if value == "apple":
                return 1
            elif value == "orange":
                return 2
            elif value == "banana":
                return 3
            elif value == 0xca11ab1e:
                return 0x5ca1ab1e + value

        @overload(dt, inline='always')
        def ol_dt(li):
            if isinstance(li, types.StringLiteral):
                value = li.literal_value
                if value == "apple":
                    def impl(li):
                        return 1
                elif value == "orange":
                    def impl(li):
                        return 2
                elif value == "banana":
                    def impl(li):
                        return 3
                return impl
            elif isinstance(li, types.IntegerLiteral):
                value = li.literal_value
                if value == 0xca11ab1e:
                    def impl(li):
                        # close over the dispatcher :)
                        return 0x5ca1ab1e + value
                    return impl

        @njit
        def foo():
            acc = 0
            for t in ('apple', 'orange', 'banana', 3390155550):
                acc += dt(t)
            return acc

        self.assertEqual(foo(), foo.py_func())

    @unittest.skip("typed.List fails")
    def test_11(self):
        # typed list fail with:
        # RuntimeError: list.append failed unexpectedly
        # gdb shows total junk in the registers for the append value
        # PartialTypeInference seems to be the root cause?!

        def foo():
            x = typed.List()
            z = ('apple', 'orange', 'banana')
            for i in range(len(z)):
                t = z[i]
                if t == "apple":
                    x.append("0")
                elif t == "orange":
                    x.append(t)
                elif t == "banana":
                    x.append("2.0")
            return x

        self.assertEqual(foo(), foo.py_func())


if __name__ == '__main__':
    unittest.main()
