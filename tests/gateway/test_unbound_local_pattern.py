"""AST regression for the UnboundLocalError family in api_server.py.

Three near-identical bugs shipped in the past week (PRs fixing
``_acquired_stream``, ``_skip_normal_call``, and ``effective_base`` —
commits 657d096b2, a8caa8e46, c2f15379e). All three had the same shape:

    def _call_xxx_passthrough(...):
        ...
        logger.info("[HTTP_LOG] foo X bar", X, ...)   # <-- reads X here
        ...
        X = (something or default)                     # <-- assigns X here

Python sees the assignment and treats the name as function-local, so the
earlier read raises UnboundLocalError.  A code review that checks "is
this variable defined before it's read?" catches the bug in seconds.
This test automates that check for ``api_server.py`` and the other large
``hermes-code`` passthrough files where the pattern keeps appearing.

The check is intentionally narrow: it only flags a *load* of a name on
a line that is *textually before* the first *unconditional* store of the
same name, in the same function.  Loads inside ``try:`` blocks before a
store that happens unconditionally in the function body are NOT flagged
(those are safe — Python only marks the name local if it sees a store
ANYWHERE in the function, but the store always runs at module import /
function call).

We allow:

* Parameter names
* Names imported at module level
* Names from ``global X`` / ``nonlocal X`` declarations
* Stores inside ``try/except`` whose corresponding load is in the
  ``except`` handler (the classic "init in try, use in except" pattern)
* Names that are only ever loaded (read-only — no function-scoped
  store at all, so no risk of UnboundLocalError)

We do NOT need to be exhaustive — even a rough filter is far better than
the status quo, which is "shipping the bug and then fixing it in a
hotfix commit three hours later".
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Files where the passthrough chain lives.  The bug has only ever appeared
# in these two so far; expand the list when the next one shows up.
TARGETS: Tuple[str, ...] = (
    "gateway/platforms/api_server.py",
    "agent/auxiliary_client.py",
)


def _line(node: ast.AST) -> int:
    return getattr(node, "lineno", 0) or 0


def _walk_store_targets(node: ast.AST) -> List[str]:
    """Yield the names bound by an assignment target.

    Handles Name, Tuple, List, Starred.
    """
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: List[str] = []
        for elt in node.elts:
            out.extend(_walk_store_targets(elt))
        return out
    if isinstance(node, ast.Starred):
        return _walk_store_targets(node.value)
    return []


# Node types that *introduce* a new scope.  Walks that hit one of these
# must NOT continue into the body — the names bound there belong to
# that scope, not the enclosing one.
_NESTED_SCOPE_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _walk_same_scope(
    node: ast.AST,
    visit,
) -> None:
    """Walk ``node`` visiting every descendant that lives in the SAME
    scope as ``node`` (skipping nested defs / lambdas / comprehensions /
    classes).  ``visit(child)`` is called for each AST node visited."""
    visit(node)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _NESTED_SCOPE_TYPES):
            # The child is a new scope — do not descend.
            continue
        _walk_same_scope(child, visit)


def _collect_function_info(source: str) -> List[Dict]:
    """For each function in the source, collect load/store info per name.

    Critically, the walk stays within the function's own scope — nested
    defs / lambdas / comprehensions have their own bindings and don't
    pollute the outer function's namespace.
    """
    tree = ast.parse(source)
    module_imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                module_imports.add(n.asname or n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                module_imports.add(n.asname or n.name)

    results: List[Dict] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params: Set[str] = set()
        for arg in (
            func.args.posonlyargs
            + func.args.args
            + func.args.kwonlyargs
        ):
            params.add(arg.arg)
        if func.args.vararg:
            params.add(func.args.vararg.arg)
        if func.args.kwarg:
            params.add(func.args.kwarg.arg)

        global_names: Set[str] = set()
        for dec in func.body:
            if isinstance(dec, (ast.Global, ast.Nonlocal)):
                global_names.update(dec.names)

        loads: Dict[str, List[int]] = {}
        stores: Dict[str, List[int]] = {}
        # Names whose ONLY stores are safe-binding stores (for-target /
        # with-as / except-as / comprehension-target) at the function's
        # own level.  Stores inside a comprehension / for-body / except
        # body are still in the function's scope (unlike defs and lambdas
        # which get their own scope), but the targets are always defined
        # before any use inside that sub-scope.
        binding_only_safe_stores: Set[str] = set()

        def visit(n: ast.AST) -> None:
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Load):
                    loads.setdefault(n.id, []).append(_line(n))
                elif isinstance(n.ctx, (ast.Store, ast.Del)):
                    stores.setdefault(n.id, []).append(_line(n))
            elif isinstance(n, ast.arg):
                pass
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                for tgt in _walk_store_targets(n.target):
                    stores.setdefault(tgt, []).append(_line(n))
                    binding_only_safe_stores.add(tgt)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars is not None:
                        for tgt in _walk_store_targets(item.optional_vars):
                            stores.setdefault(tgt, []).append(_line(n))
                            binding_only_safe_stores.add(tgt)
            elif isinstance(n, ast.ExceptHandler) and n.name is not None:
                stores.setdefault(n.name, []).append(_line(n))
                binding_only_safe_stores.add(n.name)

        _walk_same_scope(func, visit)

        # If the only stores for a name are the "always-defined-before-use"
        # binding patterns above AND there is no plain `X = ...` (Assign /
        # AnnAssign / AugAssign) to the same name at the function's own
        # level, then the safe binding is enough — no UnboundLocalError.
        for name in list(stores.keys()):
            if name in params or name in global_names or name in module_imports:
                continue
            if name not in binding_only_safe_stores:
                continue
            # Does the function have a plain `X = ...` at its own level?
            has_plain_assign = False
            for stmt in func.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, _NESTED_SCOPE_TYPES):
                        # Don't descend into a nested scope.
                        if sub is not stmt:
                            continue
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if name in _walk_store_targets(t):
                                has_plain_assign = True
                                break
                    elif isinstance(sub, ast.AnnAssign) and sub.target is not None:
                        if name in _walk_store_targets(sub.target):
                            has_plain_assign = True
                    elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Name):
                        if sub.target.id == name:
                            has_plain_assign = True
                    if has_plain_assign:
                        break
                if has_plain_assign:
                    break
            if not has_plain_assign:
                # The only stores for this name are safe bindings — drop them.
                del stores[name]

        results.append({
            "name": func.name,
            "lineno": _line(func),
            "params": params,
            "globals": global_names,
            "imports": module_imports,
            "loads": loads,
            "stores": stores,
        })
    return results


def _find_suspicious_loads(
    func_info: Dict,
) -> List[Tuple[str, int, int]]:
    """Return [(name, load_line, first_store_line), ...] for each suspicious read."""
    suspicious: List[Tuple[str, int, int]] = []
    for name, load_lines in func_info["loads"].items():
        if name in func_info["params"]:
            continue
        if name in func_info["globals"]:
            continue
        if name in func_info["imports"]:
            continue
        if name not in func_info["stores"]:
            # Pure read — no function-scoped store.  No UnboundLocalError possible.
            continue
        first_store = min(func_info["stores"][name])
        for ll in load_lines:
            if ll < first_store:
                suspicious.append((name, ll, first_store))
    return suspicious


def _format_message(file_path: str, func_info: Dict, susp: List[Tuple[str, int, int]]) -> str:
    items = ", ".join(f"{n} (load@L{ll}, first store@L{fs})" for n, ll, fs in susp)
    return (
        f"{file_path}: function `{func_info['name']}` at line {func_info['lineno']} "
        f"reads variable(s) before assigning them: {items}. "
        f"This is the UnboundLocalError pattern that has shipped three times in "
        f"the past week (see commits 657d096b2, a8caa8e46, c2f15379e). "
        f"Move the assignment above the read, or initialise the variable "
        f"with a sentinel value at the top of the function."
    )


def test_no_unbound_local_pattern_in_passthrough_files() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    seen: List[str] = []
    for rel in TARGETS:
        path = repo_root / rel
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        for func_info in _collect_function_info(source):
            susp = _find_suspicious_loads(func_info)
            if susp:
                seen.append(_format_message(rel, func_info, susp))
    assert not seen, (
        "Found UnboundLocalError-pattern loads-before-stores:\n\n"
        + "\n\n".join(seen)
    )


# ── Targeted unit tests for the checker itself ─────────────────────────
# These let us add a new false-positive suppresser without re-running the
# whole sweep against api_server.py on every commit.

class TestCheckerUnit:
    def test_load_after_store_is_fine(self) -> None:
        src = textwrap.dedent("""
            def f():
                x = 1
                return x
        """)
        infos = _collect_function_info(src)
        assert len(infos) == 1
        assert _find_suspicious_loads(infos[0]) == []

    def test_load_before_store_is_flagged(self) -> None:
        src = textwrap.dedent("""
            def f():
                logger.info("x is %s", x)
                x = 1
        """)
        infos = _collect_function_info(src)
        susp = _find_suspicious_loads(infos[0])
        assert len(susp) == 1
        name, ll, fs = susp[0]
        assert name == "x"
        assert ll < fs

    def test_parameter_read_before_reassignment_is_fine(self) -> None:
        src = textwrap.dedent("""
            def f(x):
                logger.info("x is %s", x)
                x = 1
        """)
        infos = _collect_function_info(src)
        # `x` is a parameter; the load is the parameter read, not a free var.
        # No suspicious pattern.
        assert _find_suspicious_loads(infos[0]) == []

    def test_global_load_is_fine(self) -> None:
        src = textwrap.dedent("""
            X = 0
            def f():
                logger.info("X is %s", X)
        """)
        infos = _collect_function_info(src)
        assert _find_suspicious_loads(infos[0]) == []

    def test_try_except_init_pattern_is_fine(self) -> None:
        """The classic `try: x = ...; except: ...` pattern is safe — the
        store is unconditional, the load happens after, and the only
        potential issue (the store raising) is caught by the except.  We
        don't try to flag that — it's a legit pattern."""
        src = textwrap.dedent("""
            def f():
                try:
                    x = compute()
                except Exception:
                    x = None
                return x
        """)
        infos = _collect_function_info(src)
        assert _find_suspicious_loads(infos[0]) == []

    def test_comprehension_target_is_not_a_bug(self) -> None:
        """Names like `marker` in `for marker in (...)` are
        comprehension targets — they live in the comprehension's own
        scope and don't shadow function-level names."""
        src = textwrap.dedent("""
            def f(txt):
                return any(
                    marker in txt
                    for marker in ("a", "b", "c")
                )
        """)
        infos = _collect_function_info(src)
        assert _find_suspicious_loads(infos[0]) == []

    def test_nested_function_param_does_not_pollute_outer(self) -> None:
        """A nested def's parameters and locals belong to the inner
        scope.  Reading them in the outer function is a NameError (free
        var), not an UnboundLocalError, and is therefore outside the
        scope of this check."""
        src = textwrap.dedent("""
            def outer():
                def inner(call_id, arguments):
                    if not call_id:
                        return
                    _q.put((call_id, arguments))
                inner(None, "")
        """)
        infos = _collect_function_info(src)
        # Only `outer` should be checked; `inner` is its own scope.
        outer = next(i for i in infos if i["name"] == "outer")
        assert _find_suspicious_loads(outer) == []

    def test_lambda_and_generator_handled_gracefully(self) -> None:
        # Lambdas and comprehensions are also FunctionDef/Lambda.  We
        # should not crash on them, even if we don't catch every nuance.
        src = textwrap.dedent("""
            def f():
                squares = [x*x for x in range(10)]
                return squares
        """)
        # Should not raise.
        infos = _collect_function_info(src)
        # `x` is the comprehension variable, not a bug.
        # Just check we get sensible output.
        assert any(i["name"] == "f" for i in infos)

    def test_re_assignment_later_in_function_still_flagged(self) -> None:
        """Even if the function does eventually assign the name on a
        later line, a load before the FIRST store is still a bug."""
        src = textwrap.dedent("""
            def f():
                logger.info("x is %s", x)
                if condition:
                    x = 1
                else:
                    x = 2
        """)
        infos = _collect_function_info(src)
        susp = _find_suspicious_loads(infos[0])
        assert len(susp) == 1
        assert susp[0][0] == "x"

    def test_for_loop_target_in_function_body_is_safe(self) -> None:
        """`for x in ...:` at the function level binds `x` for each
        iteration; the load inside the body always sees the bound
        value.  This is the classic safe pattern."""
        src = textwrap.dedent("""
            def f(items):
                for x in items:
                    logger.info("got %s", x)
        """)
        infos = _collect_function_info(src)
        assert _find_suspicious_loads(infos[0]) == []

    def test_caught_the_actual_recent_bug_shape(self) -> None:
        """The shape that bit us in commit c2f15379e — logger call
        referencing a name, then assignment of that name.  The fix
        swapped the two lines; the test must catch the unfixed shape."""
        src = textwrap.dedent("""
            from openai import OpenAI
            def _call_codex_passthrough(messages, model, api_key, base_url, tools=None, timeout=300):
                from agent.codex_responses_adapter import _chat_messages_to_responses_input
                logger.info("[HTTP_LOG] URL=%s model=%s", effective_base, model, api_key[:10], len(api_key))
                effective_base = (base_url or "https://chatgpt.com/backend-api/codex").rstrip("/")
                client = OpenAI(api_key=api_key, base_url=effective_base)
                return client
        """)
        infos = _collect_function_info(src)
        susp = _find_suspicious_loads(infos[0])
        assert len(susp) == 1
        assert susp[0][0] == "effective_base"
