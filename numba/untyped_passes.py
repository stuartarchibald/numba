from __future__ import print_function, division, absolute_import
from collections import defaultdict
from copy import deepcopy

from .compiler_machinery import FunctionPass, register_pass
from . import (config, bytecode, interpreter, postproc, errors, types, rewrites,
               transforms, ir, utils)
import warnings
from .analysis import (
    dead_branch_prune,
    rewrite_semantic_constants,
    find_literally_calls,
    compute_cfg_from_blocks,
    compute_use_defs,
)
from contextlib import contextmanager
from .inline_closurecall import InlineClosureCallPass, inline_closure_call
from .ir_utils import (guard, resolve_func_from_module, simplify_CFG,
                       GuardException,  convert_code_obj_to_function,
                       mk_unique_var, build_definitions,
                       replace_var_names, get_name_var_table,
                       compile_to_numba_ir,)


@contextmanager
def fallback_context(state, msg):
    """
    Wraps code that would signal a fallback to object mode
    """
    try:
        yield
    except Exception as e:
        if not state.status.can_fallback:
            raise
        else:
            if utils.PYVERSION >= (3,):
                # Clear all references attached to the traceback
                e = e.with_traceback(None)
            # this emits a warning containing the error message body in the
            # case of fallback from npm to objmode
            loop_lift = '' if state.flags.enable_looplift else 'OUT'
            msg_rewrite = ("\nCompilation is falling back to object mode "
                           "WITH%s looplifting enabled because %s"
                           % (loop_lift, msg))
            warnings.warn_explicit('%s due to: %s' % (msg_rewrite, e),
                                   errors.NumbaWarning,
                                   state.func_id.filename,
                                   state.func_id.firstlineno)
            raise


@register_pass(mutates_CFG=True, analysis_only=False)
class ExtractByteCode(FunctionPass):
    _name = "extract_bytecode"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Extract bytecode from function
        """
        func_id = state['func_id']
        bc = bytecode.ByteCode(func_id)
        if config.DUMP_BYTECODE:
            print(bc.dump())

        state['bc'] = bc
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class TranslateByteCode(FunctionPass):
    _name = "translate_bytecode"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Analyze bytecode and translating to Numba IR
        """
        func_id = state['func_id']
        bc = state['bc']
        interp = interpreter.Interpreter(func_id)
        func_ir = interp.interpret(bc)
        state["func_ir"] = func_ir
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class FixupArgs(FunctionPass):
    _name = "fixup_args"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state['nargs'] = state['func_ir'].arg_count
        if not state['args'] and state['flags'].force_pyobject:
            # Allow an empty argument types specification when object mode
            # is explicitly requested.
            state['args'] = (types.pyobject,) * state['nargs']
        elif len(state['args']) != state['nargs']:
            raise TypeError("Signature mismatch: %d argument types given, "
                            "but function takes %d arguments"
                            % (len(state['args']), state['nargs']))
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class IRProcessing(FunctionPass):
    _name = "ir_processing"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_ir = state['func_ir']
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()

        if config.DEBUG or config.DUMP_IR:
            name = func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            func_ir.dump()
            if func_ir.is_generator:
                print(("GENERATOR INFO: %s" % name).center(80, "-"))
                func_ir.dump_generator_info()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class RewriteSemanticConstants(FunctionPass):
    _name = "rewrite_semantic_constants"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
        assert state.func_ir
        msg = ('Internal error in pre-inference dead branch pruning '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            rewrite_semantic_constants(state.func_ir, state.args)

        if config.DEBUG or config.DUMP_IR:
            print('branch_pruned_ir'.center(80, '-'))
            print(state.func_ir.dump())
            print('end branch_pruned_ir'.center(80, '-'))
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DeadBranchPrune(FunctionPass):
    _name = "dead_branch_prune"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """

        # purely for demonstration purposes, obtain the analysis from a pass
        # declare as a required dependent
        semantic_const_analysis = self.get_analysis(type(self))  # noqa

        assert state.func_ir
        msg = ('Internal error in pre-inference dead branch pruning '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            dead_branch_prune(state.func_ir, state.args)

        if config.DEBUG or config.DUMP_IR:
            print('branch_pruned_ir'.center(80, '-'))
            print(state.func_ir.dump())
            print('end branch_pruned_ir'.center(80, '-'))

        return True

    def get_analysis_usage(self, AU):
        AU.add_required(RewriteSemanticConstants)


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineClosureLikes(FunctionPass):
    _name = "inline_closure_likes"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # Ensure we have an IR and type information.
        assert state.func_ir

        # if the return type is a pyobject, there's no type info available and
        # no ability to resolve certain typed function calls in the array
        # inlining code, use this variable to indicate
        typed_pass = not isinstance(state.return_type, types.misc.PyObject)
        inline_pass = InlineClosureCallPass(
            state.func_ir,
            state.flags.auto_parallel,
            state.parfor_diagnostics.replaced_fns,
            typed_pass)
        inline_pass.run()
        # Remove all Dels, and re-run postproc
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class GenericRewrites(FunctionPass):
    _name = "generic_rewrites"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Perform any intermediate representation rewrites before type
        inference.
        """
        assert state.func_ir
        msg = ('Internal error in pre-inference rewriting '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            rewrites.rewrite_registry.apply('before-inference', state)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class WithLifting(FunctionPass):
    _name = "with_lifting"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Extract with-contexts
        """
        main, withs = transforms.with_lifting(
            func_ir=state.func_ir,
            typingctx=state.typingctx,
            targetctx=state.targetctx,
            flags=state.flags,
            locals=state.locals,
        )
        if withs:
            from numba.compiler import compile_ir, _EarlyPipelineCompletion
            cres = compile_ir(state.typingctx, state.targetctx, main,
                              state.args, state.return_type,
                              state.flags, state.locals,
                              lifted=tuple(withs), lifted_from=None,
                              pipeline_class=type(state.pipeline))
            raise _EarlyPipelineCompletion(cres)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineInlinables(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.jit decorator directly
    into the site of its call depending on the value set in the 'inline' kwarg
    to the decorator.

    This is an untyped pass. CFG simplification is performed at the end of the
    pass but no block level clean up is performed on the mutated IR (typing
    information is not available to do so).
    """
    _name = "inline_inlinables"
    _DEBUG = False

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """Run inlining of inlinables
        """
        if config.DEBUG or self._DEBUG:
            print('before inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        modified = False
        # use a work list, look for call sites via `ir.Expr.op == call` and
        # then pass these to `self._do_work` to make decisions about inlining.
        work_list = list(state.func_ir.blocks.items())
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        if guard(self._do_work, state, work_list, block, i,
                                 expr):
                            modified = True
                            break  # because block structure changed

        if modified:
            # clean up unconditional branches that appear due to inlined
            # functions introducing blocks
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        if config.DEBUG or self._DEBUG:
            print('after inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        return True

    def _do_work(self, state, work_list, block, i, expr):
        from numba.inline_closurecall import (inline_closure_call,
                                              callee_ir_validator)
        from numba.compiler import run_frontend
        from numba.targets.cpu import InlineOptions

        # try and get a definition for the call, this isn't always possible as
        # it might be a eval(str)/part generated awaiting update etc. (parfors)
        to_inline = None
        try:
            to_inline = state.func_ir.get_definition(expr.func)
        except Exception:
            if self._DEBUG:
                print("Cannot find definition for %s" % expr.func)
            return False
        # do not handle closure inlining here, another pass deals with that.
        if getattr(to_inline, 'op', False) == 'make_function':
            return False

        # see if the definition is a "getattr", in which case walk the IR to
        # try and find the python function via the module from which it's
        # imported, this should all be encoded in the IR.
        if getattr(to_inline, 'op', False) == 'getattr':
            val = resolve_func_from_module(state.func_ir, to_inline)
        else:
            # This is likely a freevar or global
            #
            # NOTE: getattr 'value' on a call may fail if it's an ir.Expr as
            # getattr is overloaded to look in _kws.
            try:
                val = getattr(to_inline, 'value', False)
            except Exception:
                raise GuardException

        # if something was found...
        if val:
            # check it's dispatcher-like, the targetoptions attr holds the
            # kwargs supplied in the jit decorator and is where 'inline' will
            # be if it is present.
            topt = getattr(val, 'targetoptions', False)
            if topt:
                inline_type = topt.get('inline', None)
                # has 'inline' been specified?
                if inline_type is not None:
                    inline_opt = InlineOptions(inline_type)
                    # Could this be inlinable?
                    if not inline_opt.is_never_inline:
                        # yes, it could be inlinable
                        do_inline = True
                        pyfunc = val.py_func
                        # Has it got an associated cost model?
                        if inline_opt.has_cost_model:
                            # yes, it has a cost model, use it to determine
                            # whether to do the inline
                            py_func_ir = run_frontend(pyfunc)
                            do_inline = inline_type(expr, state.func_ir,
                                                    py_func_ir)
                        # if do_inline is True then inline!
                        if do_inline:
                            inline_closure_call(
                                state.func_ir,
                                pyfunc.__globals__,
                                block, i, pyfunc,
                                work_list=work_list,
                                callee_validator=callee_ir_validator)
                            return True
        return False


@register_pass(mutates_CFG=False, analysis_only=False)
class PreserveIR(FunctionPass):
    """
    Preserves the IR in the metadata
    """

    _name = "preserve_ir"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.metadata['preserved_ir'] = state.func_ir.copy()
        return False


@register_pass(mutates_CFG=False, analysis_only=True)
class FindLiterallyCalls(FunctionPass):
    """Find calls to `numba.literally()` and signal if its requirement is not
    satisfied.
    """
    _name = "find_literally"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        find_literally_calls(state.func_ir, state.args)
        return False


@register_pass(mutates_CFG=True, analysis_only=False)
class MakeFunctionToJitFunction(FunctionPass):
    """
    This swaps an ir.Expr.op == "make_function" i.e. a closure, for a compiled
    function containing the closure body and puts it in ir.Global. It's a 1:1
    statement value swap. `make_function` is already untyped
    """
    _name = "make_function_op_code_to_jit_function"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        from numba import njit
        func_ir = state.func_ir
        mutated = False
        for idx, blk in func_ir.blocks.items():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Expr):
                        if stmt.value.op == "make_function":
                            node = stmt.value
                            getdef = func_ir.get_definition
                            kw_default = getdef(node.defaults)
                            ok = False
                            if (kw_default is None or
                                    isinstance(kw_default, ir.Const)):
                                ok = True
                            elif isinstance(kw_default, tuple):
                                ok = all([isinstance(getdef(x), ir.Const)
                                          for x in kw_default])

                            if not ok:
                                continue

                            pyfunc = convert_code_obj_to_function(node, func_ir)
                            func = njit()(pyfunc)
                            new_node = ir.Global(node.code.co_name, func,
                                                 stmt.loc)
                            stmt.value = new_node
                            mutated |= True

        # if a change was made the del ordering is probably wrong, patch up
        if mutated:
            post_proc = postproc.PostProcessor(func_ir)
            post_proc.run()

        return mutated


@register_pass(mutates_CFG=True, analysis_only=False)
class MixedContainerUnroller(FunctionPass):
    _name = "mixed_container_unroller"

    _DEBUG = False

    def __init__(self):
        FunctionPass.__init__(self)

    def analyse_tuple(self, tup):
        d = defaultdict(list)
        for i, ty in enumerate(tup):
            d[ty].append(i)
        return d

    def add_offset_to_labels_w_ignore(self, blocks, offset, ignore=None):
        """add an offset to all block labels and jump/branch targets
        don't add an offset to anything in the ignore list
        """
        if ignore is None:
            ignore = set()

        new_blocks = {}
        for l, b in blocks.items():
            # some parfor last blocks might be empty
            term = None
            if b.body:
                term = b.body[-1]
            if isinstance(term, ir.Jump):
                if term.target not in ignore:
                    b.body[-1] = ir.Jump(term.target + offset, term.loc)
            if isinstance(term, ir.Branch):
                if term.truebr not in ignore:
                    new_true = term.truebr + offset
                else:
                    new_true = term.truebr

                if term.falsebr not in ignore:
                    new_false = term.falsebr + offset
                else:
                    new_false = term.falsebr
                b.body[-1] = ir.Branch(term.cond, new_true, new_false, term.loc)
            new_blocks[l + offset] = b
        return new_blocks

    def stuff_in_loop_body(self, func_ir, loop_ir, caller_max_label,
                           dont_replace, switch_data):

        # Find the sentinels and validate the form
        sentinel_exits = set()
        sentinel_blocks = []
        for lbl, blk in func_ir.blocks.items():
            for i, stmt in enumerate(blk.body):
                if isinstance(stmt, ir.Assign):
                    if "SENTINEL" in stmt.target.name:
                        sentinel_blocks.append(lbl)
                        sentinel_exits.add(blk.body[-1].target)
                        break

        assert len(sentinel_exits) == 1  # should only be 1 exit
        func_ir.blocks.pop(sentinel_exits.pop())  # kill the exit, it's dead

        # find jumps that are non-local, we won't relabel these
        ignore_set = set()
        local_lbl = [x for x in loop_ir.blocks.keys()]
        for lbl, blk in loop_ir.blocks.items():
            for i, stmt in enumerate(blk.body):
                if isinstance(stmt, ir.Jump):
                    if stmt.target not in local_lbl:
                        ignore_set.add(stmt.target)
                if isinstance(stmt, ir.Branch):
                    if stmt.truebr not in local_lbl:
                        ignore_set.add(stmt.truebr)
                    if stmt.falsebr not in local_lbl:
                        ignore_set.add(stmt.falsebr)

        # make sure the generated switch table matches the switch data
        assert len(sentinel_blocks) == len(switch_data)

        # replace the sentinel_blocks with the loop body
        for lbl, branch_ty in zip(sentinel_blocks, switch_data.keys()):
            loop_blocks = deepcopy(loop_ir.blocks)
            # relabel blocks
            max_label = max(func_ir.blocks.keys())
            loop_blocks = self.add_offset_to_labels_w_ignore(
                loop_blocks, max_label + 1, ignore_set)

            # start label
            loop_start_lbl = min(loop_blocks.keys())

            # fix the typed_getitem locations in the loop blocks
            for blk in loop_blocks.values():
                new_body = []
                for stmt in blk.body:
                    if isinstance(stmt, ir.Assign):
                        if (isinstance(stmt.value, ir.Expr) and
                                stmt.value.op == "typed_getitem"):
                            if isinstance(branch_ty, types.Literal):
                                new_const_name = mk_unique_var("branch_const")
                                new_const_var = ir.Var(
                                    blk.scope, new_const_name, stmt.loc)
                                new_const_val = ir.Const(
                                    branch_ty.literal_value, stmt.loc)
                                const_assign = ir.Assign(
                                    new_const_val, new_const_var, stmt.loc)
                                new_assign = ir.Assign(
                                    new_const_var, stmt.target, stmt.loc)
                                new_body.append(const_assign)
                                new_body.append(new_assign)
                                dont_replace.append(new_const_name)
                            else:
                                orig = stmt.value
                                new_typed_getitem = ir.Expr.typed_getitem(
                                    value=orig.value, dtype=branch_ty,
                                    index=orig.index, loc=orig.loc)
                                new_assign = ir.Assign(
                                    new_typed_getitem, stmt.target, stmt.loc)
                                new_body.append(new_assign)
                                pass
                        else:
                            new_body.append(stmt)
                    else:
                        new_body.append(stmt)
                blk.body = new_body

            # rename
            var_table = get_name_var_table(loop_blocks)
            drop_keys = []
            for k, v in var_table.items():
                if v.name in dont_replace:
                    drop_keys.append(k)
            for k in drop_keys:
                var_table.pop(k)

            new_var_dict = {}
            for name, var in var_table.items():
                new_var_dict[name] = mk_unique_var(name)
            replace_var_names(loop_blocks, new_var_dict)

            # clobber the sentinel body and then stuff in the rest
            func_ir.blocks[lbl] = deepcopy(loop_blocks[loop_start_lbl])
            remaining_keys = [y for y in loop_blocks.keys()]
            remaining_keys.remove(loop_start_lbl)
            for k in remaining_keys:
                func_ir.blocks[k] = deepcopy(loop_blocks[k])

        # now relabel the func_ir WRT the caller max label
        func_ir.blocks = self.add_offset_to_labels_w_ignore(
            func_ir.blocks, caller_max_label + 1, ignore_set)

        if self._DEBUG:
            print("-" * 80 + "EXIT STUFFER")
            func_ir.dump()
            print("-" * 80)

        return func_ir

    def gen_switch(self, data, index):
        """
        Generates a function with a switch table like
        def foo():
            if PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            elif PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            ...
            else:
                raise RuntimeError

        The data is a map of (type : indexes) for example:
        (int64, int64, float64)
        might give:
        {int64: [0, 1], float64: [2]}

        The index is the index variable for the driving range loop over the
        mixed tuple.
        """
        elif_tplt = "\n\telif PLACEHOLDER_INDEX in (%s,):\n\t\tSENTINEL = None"

        b = ('def foo():\n\tif PLACEHOLDER_INDEX in (%s,):\n\t\t'
             'SENTINEL = None\n%s\n\telse:\n\t\t'
             'raise RuntimeError("Unreachable")')
        keys = [k for k in data.keys()]

        elifs = []
        for i in range(1, len(keys)):
            elifs.append(elif_tplt % ','.join(map(str, data[keys[i]])))
        src = b % (','.join(map(str, data[keys[0]])), ''.join(elifs))
        wstr = src
        l = {}
        exec(wstr, {}, l)
        bfunc = l['foo']
        branches = compile_to_numba_ir(bfunc, {})
        for lbl, blk in branches.blocks.items():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Global):
                        if stmt.value.name == "PLACEHOLDER_INDEX":
                            stmt.value = index
        return branches

    def run_pass(self, state):
        mutated = False
        func_ir = state.func_ir
        # first limit the work by squashing the CFG if possible
        func_ir.blocks = simplify_CFG(func_ir.blocks)

        if self._DEBUG:
            print("-" * 80 + "PASS ENTRY")
            func_ir.dump()
            print("-" * 80)

        # compute new CFG
        cfg = compute_cfg_from_blocks(func_ir.blocks)
        # find loops, usedefs, liveness etc
        loops = cfg.loops()
        usedefs = compute_use_defs(func_ir.blocks)
        smash = dict()
        for lbl, blk in state.func_ir.blocks.items():
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value,
                                  ir.Expr) and stmt.value.op == "getitem":
                        getitem_target = stmt.value.value
                        target_ty = state.typemap[getitem_target.name]
                        if not isinstance(target_ty, types.Tuple):
                            continue

                        # get switch data
                        switch_data = self.analyse_tuple(target_ty)

                        # generate switch IR
                        index = func_ir._definitions[stmt.value.index.name][0]
                        branches = self.gen_switch(switch_data, index)

                        # swap getitems for a typed_getitem, these are actually
                        # just placeholders at this point. When the loop is
                        # duplicated they can be swapped for a typed_getitem of
                        # the correct type or if the item is literal it can be
                        # shoved straight into the duplicated loop body
                        old = stmt.value
                        new = ir.Expr.typed_getitem(
                            old.value, types.void, old.index, old.loc)
                        stmt.value = new

                        # find the loops
                        for l in loops.values():
                            if lbl in l.body:
                                this_loop = l
                                break

                        assert this_loop
                        this_loop_body = this_loop.body - \
                            set([this_loop.header])
                        loop_blocks = {
                            x: func_ir.blocks[x] for x in this_loop_body}

                        # get some new IR based on the original, but just
                        # comprising the loop blocks
                        new_ir = func_ir.derive(loop_blocks)

                        # anything used/defined in the body or is live at the
                        # loop header shouldn't be messed with
                        idx = this_loop.header
                        keep = set()
                        keep |= usedefs.usemap[idx] | usedefs.defmap[idx]
                        keep |= func_ir.variable_lifetime.livemap[idx]
                        dont_replace = [x for x in (keep)]

                        unrolled_body = self.stuff_in_loop_body(
                            branches, new_ir, max(func_ir.blocks.keys()),
                            dont_replace, switch_data)
                        smash[tuple(this_loop_body)] = (
                            unrolled_body, this_loop.header)

                        mutated |= True

        blks = state.func_ir.blocks
        for orig_lbl, data in smash.items():
            replace, *delete = orig_lbl
            unroll, header_block = data
            unroll_lbl = [x for x in sorted(unroll.blocks.keys())]
            blks[replace] = unroll.blocks[unroll_lbl[0]]
            [blks.pop(d) for d in delete]
            for k in unroll_lbl[1:]:
                blks[k] = unroll.blocks[k]
            # stitch up the loop predicate true -> new loop body jump
            blks[header_block].body[-1].truebr = replace
        if self._DEBUG:
            print('-' * 80 + "END OF PASS, DOING SIMPLIFY")
            func_ir.dump()
        func_ir.blocks = simplify_CFG(func_ir.blocks)
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()
        if self._DEBUG:
            print('-' * 80 + "END OF PASS, SIMPLIFY DONE")
            func_ir.dump()
        # rebuild the definitions table, the IR has taken a hammering
        func_ir._definitions = build_definitions(func_ir.blocks)
        # reset type inference now we are done with the partial results
        state.typemap = {}
        state.return_type = None
        return mutated


@register_pass(mutates_CFG=True, analysis_only=False)
class IterLoopCanonicalization(FunctionPass):
    _name = "iter_loop_canonicalisation"

    _DEBUG = False

    _accepted_types = (types.Tuple, types.UniTuple)

    def __init__(self):
        FunctionPass.__init__(self)


    def assess_loop(self, loop, func_ir, partial_typemap):
        # it's a iter loop if:
        # - loop header is driven by an iternext
        # - the iternext value is a phi derived from getiter()

        # check header
        iternexts = [*func_ir.blocks[loop.header].find_exprs('iternext')]
        if len(iternexts) != 1: return False
        for iternext in iternexts:
            phi = func_ir.get_definition(iternext.value)
            if getattr(phi, 'op', False) == 'getiter':
                ty = partial_typemap.get(phi.value.name, None)
                if ty and isinstance(ty, self._accepted_types):
                    return len(loop.entries) == 1

    def mangle(self, loop, func_ir, cfg):
        def get_range(a):
            return range(len(a))

        iternext = [*func_ir.blocks[loop.header].find_exprs('iternext')][0]
        LOC=func_ir.blocks[loop.header].loc
        from numba.ir_utils import mk_unique_var
        get_range_var = ir.Var(func_ir.blocks[loop.header].scope, mk_unique_var('get_range_gbl'), LOC)
        get_range_global = ir.Global('get_range', get_range, LOC)
        assgn = ir.Assign(get_range_global, get_range_var, LOC)

        loop_entry = tuple(loop.entries)[0]
        entry_block = func_ir.blocks[loop_entry]
        entry_block.body.insert(0, assgn)

        iterarg = func_ir.get_definition(iternext.value).value

        # look for iternext
        idx = 0
        for stmt in entry_block.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'getiter':
                    break
            idx += 1
        else:
            raise ValueError("problem")

        # create a range(len(tup)) and inject it
        call_get_range_var = ir.Var(entry_block.scope,
                                    mk_unique_var('call_get_range'), LOC)
        make_call = ir.Expr.call(get_range_var, (stmt.value.value,), (), LOC)
        assgn_call = ir.Assign(make_call, call_get_range_var, LOC)
        entry_block.body.insert(idx, assgn_call)
        entry_block.body[idx + 1].value.value = call_get_range_var

        f = compile_to_numba_ir(get_range, {})
        import copy
        glbls = copy.copy(func_ir.func_id.func.__globals__)
        inline_closure_call(func_ir, glbls, entry_block, idx, get_range,)
        kill = entry_block.body.index(assgn)
        entry_block.body.pop(kill)

        # find the induction variable + references in the loop header
        # fixed point iter to do this, it's a bit clunky
        induction_vars = set()
        header_block = func_ir.blocks[loop.header]

        # find induction var
        ind = [x for x in header_block.find_exprs('pair_first')]
        for x in ind:
            induction_vars.add(func_ir.get_assignee(x, loop.header))
        # find aliases of the induction var
        tmp = set()
        for x in induction_vars:
            tmp.add(func_ir.get_assignee(x, loop.header))
        induction_vars |= tmp

        # Find the downstream blocks that might reference the induction var
        succ = set()
        for lbl in loop.exits:
            succ |= set([x[0] for x in cfg.successors(lbl)])
        check_blocks = (loop.body | loop.exits | succ) ^ {loop.header}

        # replace RHS use of induction var with getitem
        for lbl in check_blocks:
            for stmt in func_ir.blocks[lbl].body:
                if isinstance(stmt, ir.Assign):
                    if stmt.value in induction_vars:
                        stmt.value = ir.Expr.getitem(iterarg, stmt.value, stmt.loc)

        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()

    def run_pass(self, state):
        func_ir = state.func_ir
        cfg = compute_cfg_from_blocks(func_ir.blocks)
        loops = cfg.loops()

        mutated = False
        accepted_loops = []
        for header, loop in loops.items():
            if self.assess_loop(loop, func_ir, state.typemap):
                self.mangle(loop, func_ir, cfg)
                mutated = True

        func_ir.blocks = simplify_CFG(func_ir.blocks)
        return mutated
