"""Admission lint: a catalog plugin must not rebind Hermes core at runtime.

Plugins extend Hermes through public surfaces (hooks, middleware, provider profiles, Desktop SDK
slots). A plugin that replaces a core function, method or module attribute in place — assigning
``AIAgent._replace_primary_openai_client``, ``tui_gateway.server.handle_request``, a
``gateway.run_turn`` helper, a private catalog dict — collides with every other plugin that patches
the same seam and breaks on any core release. This static pass finds those rebinds in the plugin's
Python (the import closure of ``__init__.py`` and ``dashboard/``; tests, benchmarks and scripts
excluded). Like the Desktop lint it is a review tripwire, not a sandbox.

A value is "core" when it is a Hermes module or an attribute chain off one: a name bound by an
absolute import of a Hermes top-level package, ``importlib.import_module(...)`` / ``sys.modules``
lookups, ``getattr(core, ...)``, iteration over core values, and calls to plugin functions that
return a core value. Rebinds are attribute assignment / deletion, ``setattr`` / ``delattr``,
``sys.modules[...] = ...``, ``mock.patch``, and passing a core value to a plugin helper that
``setattr``s its parameter.

In-place mutation of a core object's PRIVATE state is a rebind too (``get_plugin_manager()._hooks[
"x"].append(fn)``, ``mgr._middleware.clear()``, ``mgr._hooks[:] = []``). For these sinks a value
returned by a core function also counts as core, but only a receiver chain with a private segment
is judged, so calling a public method on a core object (``registry.add(...)``) stays clean.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_SKIP_DIRS = frozenset({"tests", "test", "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages"})


@lru_cache(maxsize=1)
def _hermes_top_level() -> frozenset:
    """Top-level importable names shipped by this Hermes checkout (packages and root modules)."""
    root = Path(__file__).resolve().parents[1]
    names = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    names |= {p.stem for p in root.glob("*.py")}
    # ``hermes_plugins.*`` is where the loader imports plugins (bundled platform adapters included):
    # rebinding another plugin's module is the same collision as rebinding core.
    return frozenset((names | {"hermes_plugins"}) - {"setup", "conftest"})


def _is_test_file(rel: Path) -> bool:
    name = rel.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _python_files(plugin_dir: Path) -> Iterable[tuple[Path, Path]]:
    for path in sorted(plugin_dir.rglob("*.py")):
        rel = path.relative_to(plugin_dir)
        if _SKIP_DIRS.intersection(rel.parts[:-1]) or _is_test_file(rel):
            continue
        yield path, rel


def _runtime_files(plugin_dir: Path, files: dict[Path, ast.AST], local: set[str]) -> set[Path]:
    """The import closure of what Hermes loads: ``__init__.py`` (``register``) and ``dashboard/*.py``
    (dashboard plugin API). Benchmarks, CI and smoke scripts shipped beside them never run inside
    Hermes, so a stub they install into ``sys.modules`` is not a runtime override."""
    by_module: dict[str, set[Path]] = {}
    for rel in files:
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for start in range(len(parts)):
            by_module.setdefault(".".join(parts[start:]), set()).add(rel)

    def resolve(rel: Path, node: ast.AST) -> set[Path]:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = rel.parent.parts[: len(rel.parent.parts) - (node.level - 1)] if node.level > 1 else rel.parent.parts
                prefix = ".".join(base + tuple((node.module or "").split("."))).strip(".")
                names = [prefix] + [f"{prefix}.{a.name}".strip(".") for a in node.names]
            elif node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        found: set[Path] = set()
        for name in names:
            if name.split(".")[0] in local or isinstance(node, ast.ImportFrom) and node.level:
                found |= by_module.get(name, set())
        return found

    roots = {rel for rel in files if rel == Path("__init__.py") or (len(rel.parts) == 2 and rel.parts[0] == "dashboard")}
    seen: set[Path] = set()
    stack = list(roots)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for node in ast.walk(files[rel]):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                stack.extend(resolve(rel, node) - seen)
    return seen


def _local_names(plugin_dir: Path) -> set[str]:
    """Module/package names the plugin ships itself: an import of these is local, not core."""
    names = set()
    for path in plugin_dir.rglob("*"):
        if _SKIP_DIRS.intersection(path.relative_to(plugin_dir).parts):
            continue
        if path.suffix == ".py":
            names.add(path.stem)
        elif path.is_dir():
            names.add(path.name)
    return names


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_name(node: ast.Call) -> str:
    return _dotted(node.func)


_BUILTIN_TYPES = frozenset({"set", "dict", "list", "frozenset", "tuple", "str", "bytes", "bytearray",
                            "deque", "defaultdict", "OrderedDict", "Counter"})

# Methods that change a list / dict / set in place.
_MUTATORS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "update", "setdefault", "add", "discard", "popitem"})


def _render(node: ast.AST) -> str:
    """``get_plugin_manager()._hooks[...]``: a receiver chain as the finding names it."""
    if isinstance(node, ast.Attribute):
        return f"{_render(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_render(node.value)}[...]"
    if isinstance(node, ast.Call):
        return f"{_render(node.func)}()"
    return node.id if isinstance(node, ast.Name) else "<core>"


def _has_mutator_call(node: ast.AST) -> bool:
    """Whether the chain already holds a mutator call (``m._hooks.setdefault(k, []).append``):
    that inner call is the finding, the outer one would repeat it."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
            return True
        node = node.func if isinstance(node, ast.Call) else node.value
    return False


def _has_private_segment(node: ast.AST) -> bool:
    """Whether an attribute in the chain (``mgr._hooks[...]``, ``m.__dict__``) is private."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return True
        node = node.func if isinstance(node, ast.Call) else node.value
    return False


class _Analysis:
    def __init__(self, core_roots: frozenset, local: set[str]):
        self.core_roots = core_roots - local
        self.local = local
        self.core_returners: set[str] = set()
        # Helpers that rebind / mutate a parameter, keyed by ``name`` for a plain function and by
        # ``(class, name)`` for a method: ``set.add(x)`` is not the plugin's own ``Recorder.add``.
        self.param_patchers: dict[object, set[int]] = {}
        self.param_mutators: dict[object, set[int]] = {}  # helpers mutating a parameter's private state
        self.aliases: dict[str, str] = {}  # ``from .compat import server as gateway_server``
        self.constants: dict[str, str] = {}  # module-level ``NAME = "module.path"``
        self.class_bases: dict[str, set[str]] = {}  # the plugin's classes and their plugin bases
        self.instances: dict[str, set] = {}  # ``self.<attr>`` slots and what they are bound to
        # The scope being judged: its class (``self``) and its local / module name bindings.
        self.scope_cls: str | None = None
        self.bindings: dict[str, set] = {}
        self.module_bindings: dict[str, set] = {}

    def _func_key(self, call: ast.Call) -> str:
        name = _call_name(call).split(".")[-1]
        return self.aliases.get(name, name)

    def _class_of_call(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = self.aliases.get(node.func.id, node.func.id)
            return name if name in self.class_bases else None
        return None

    def _kinds(self, value: ast.AST) -> set:
        """What a bound value is, for resolving a later method call on it: a plugin class
        (``Patches()``), ``"builtin"`` (a container literal or builtin constructor, whose methods
        are never the plugin's) or ``"other"`` (anything else: a factory, a parameter)."""
        cls = self._class_of_call(value)
        if cls:
            return {cls}
        if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.Constant, ast.JoinedStr,
                              ast.DictComp, ast.ListComp, ast.SetComp)):
            return {"builtin"}
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in _BUILTIN_TYPES:
            return {"builtin"}
        return {"other"}

    def scope_bindings(self, body: list[ast.stmt], params: Iterable[str] = ()) -> dict[str, set]:
        bindings: dict[str, set] = {p: {"other"} for p in params}
        for node in (n for stmt in body for n in _walk_scope(stmt)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                plain = [t for t in targets if isinstance(t, ast.Name)]
                for t in plain:
                    bindings.setdefault(t.id, set()).update(self._kinds(node.value))
                for name in _bound_names([t for t in targets if not isinstance(t, ast.Name)]):
                    bindings.setdefault(name, set()).add("other")
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension, ast.With, ast.AsyncWith)):
                targets = [node.target] if not isinstance(node, (ast.With, ast.AsyncWith)) else \
                    [i.optional_vars for i in node.items if i.optional_vars is not None]
                for name in _bound_names(targets):
                    bindings.setdefault(name, set()).add("other")
        return bindings

    def _receiver_classes(self, recv: ast.AST, tainted: set[str]) -> set[str] | None:
        """Plugin classes (with their bases) a method call's receiver may be an instance of, or
        ``None`` when the receiver cannot be resolved (then any same-named helper may be meant)."""
        if self.is_core(recv, tainted):
            return set()  # a method on a core object is core's own, never the plugin's helper
        kinds = self._kinds(recv) if isinstance(recv, ast.Call) else \
            {"builtin"} if isinstance(recv, (ast.Dict, ast.List, ast.Set, ast.Constant)) else None
        if isinstance(recv, ast.Name):
            kinds = {self.scope_cls} if recv.id in ("self", "cls") and self.scope_cls else \
                self.bindings.get(recv.id) or self.module_bindings.get(recv.id)
        elif isinstance(recv, ast.Attribute) and isinstance(recv.value, ast.Name) and recv.value.id == "self":
            kinds = self.instances.get("." + recv.attr)
        if not kinds or "other" in kinds:
            return None
        found = kinds - {"builtin"}
        stack = list(found)
        while stack:
            for base in self.class_bases.get(stack.pop(), ()):
                if base not in found:
                    found.add(base)
                    stack.append(base)
        return found

    def helper_indices(self, table: dict, call: ast.Call, tainted: set[str]) -> set[int]:
        """Argument positions a call hands to a patching / mutating helper. A plain call
        (``_wrap(...)``, ``compat._wrap(...)``) matches the function by name; a method call
        matches the method of the receiver's plugin class, and only an unresolvable receiver
        falls back to any helper of that name (``set().add(x)`` is not ``Recorder.add``)."""
        func = call.func
        if not isinstance(func, ast.Attribute) or isinstance(func.value, ast.Name) and func.value.id in self.local:
            return set(table.get(self._func_key(call), ()))
        classes = self._receiver_classes(func.value, tainted)
        if classes is None:
            return set(table.get(func.attr, ())).union(
                *(v for k, v in table.items() if isinstance(k, tuple) and k[1] == func.attr))
        return {i for cls in classes for i in table.get((cls, func.attr), ())}

    # -- what counts as a core value -------------------------------------------------------------
    def _core_module_name(self, name: object) -> bool:
        return isinstance(name, str) and name.split(".")[0] in self.core_roots

    def is_core(self, node: ast.AST, tainted: set[str], calls: bool = False) -> bool:
        """``calls``: a call to a core function returns core too (the private-state sinks only)."""
        if isinstance(node, ast.Name):
            return node.id in tainted
        if isinstance(node, ast.Attribute):
            return self.is_core(node.value, tainted, calls)
        if isinstance(node, ast.Subscript):
            if _dotted(node.value) == "sys.modules":
                return self._lookup_is_core(node.slice)
            return self.is_core(node.value, tainted, calls)
        if isinstance(node, ast.Call):
            if calls and self.is_core(node.func, tainted, calls):
                return True
            name = _call_name(node)
            if name in ("importlib.import_module", "import_module", "__import__") and node.args:
                return self._lookup_is_core(node.args[0])
            if name == "sys.modules.get" and node.args:
                return self._lookup_is_core(node.args[0])
            if name == "getattr" and node.args:
                return self.is_core(node.args[0], tainted)
            return self._func_key(node) in self.core_returners
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(self.is_core(e, tainted, calls) for e in node.elts)
        return False

    def _lookup_is_core(self, arg: ast.AST) -> bool:
        """Whether a module-name expression names Hermes: a literal, a module-level string
        constant, or an f-string whose literal head names it (``f"hermes_plugins.{p}.adapter"``).
        A name computed some other way (the plugin locating its own package) is not judged."""
        if isinstance(arg, ast.Name):
            arg = ast.Constant(self.constants.get(arg.id))
        if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            head = str(arg.values[0].value)
            return "." in head and self._core_module_name(head)
        return isinstance(arg, ast.Constant) and self._core_module_name(arg.value)

    def _private_core_state(self, node: ast.AST, derived: set[str]) -> bool:
        return _has_private_segment(node) and self.is_core(node, derived, calls=True)

    def private_mutations(self, node: ast.AST, derived: set[str]) -> list[str]:
        """In-place writes to a core object's private state: a mutator call, or an item / slice
        assignment or delete, on a private receiver chain."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _MUTATORS \
                    and self._private_core_state(func.value, derived) and not _has_mutator_call(func.value):
                return [f"{_render(func)}(...)"]
            name = _call_name(node)
            return [f"{name}({_dotted(node.args[i]) or '<core>'}, ...)"
                    for i in sorted(self.helper_indices(self.param_mutators, node, derived))
                    if i < len(node.args) and self.is_core(node.args[i], derived, calls=True)]
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else \
            [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        return [_render(t) for target in targets
                for t in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])
                if isinstance(t, ast.Subscript) and self._private_core_state(t, derived)]

    # -- per-scope taint --------------------------------------------------------------------------
    def scope_taint(self, body: list[ast.stmt], inherited: set[str], calls: bool = False) -> set[str]:
        tainted = set(inherited)
        nodes = [n for stmt in body for n in _walk_scope(stmt)]
        for _ in range(4):
            before = len(tainted)
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._core_module_name(alias.name):
                            tainted.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and self._core_module_name(node.module):
                    tainted.update(a.asname or a.name for a in node.names)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    if self.is_core(node.value, tainted, calls):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        tainted.update(_bound_names(targets))
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    if self.is_core(node.iter, tainted, calls):
                        tainted.update(_bound_names([node.target]))
            if len(tainted) == before:
                break
        return tainted

    # -- sinks ------------------------------------------------------------------------------------
    def rebinds(self, node: ast.AST, tainted: set[str], derived: set[str] | None = None) -> list[str]:
        """``derived``: the scope's taint with core-function returns included (``scope_taint(calls=True)``)."""
        hits: list[str] = []
        targets: Sequence[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            for t in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                if isinstance(t, ast.Attribute) and self.is_core(t.value, tainted):
                    hits.append(_dotted(t) or t.attr)
                elif isinstance(t, ast.Subscript) and _dotted(t.value) == "sys.modules":
                    if isinstance(t.slice, ast.Constant) and self._core_module_name(t.slice.value):
                        hits.append(f"sys.modules[{t.slice.value!r}]")
                elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
                        and self.is_core(t.value.value, tainted):
                    hits.append(f"{_dotted(t.value) or t.value.attr}[...]")
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in ("setattr", "delattr") and node.args and self.is_core(node.args[0], tainted):
                hits.append(f"{name}({_dotted(node.args[0]) or '<core>'}, ...)")
            elif name.endswith("patch.object") and node.args and self.is_core(node.args[0], tainted):
                hits.append(f"patch.object({_dotted(node.args[0])}, ...)")
            elif name.split(".")[-1] == "patch" and node.args and isinstance(node.args[0], ast.Constant) \
                    and self._core_module_name(node.args[0].value):
                hits.append(f"patch({node.args[0].value!r})")
            else:
                for index in sorted(self.helper_indices(self.param_patchers, node, tainted)):
                    if index < len(node.args) and self.is_core(node.args[index], tainted):
                        hits.append(f"{name}({_dotted(node.args[index]) or '<core>'}, ...)")
        return hits + [h for h in self.private_mutations(node, tainted if derived is None else derived)
                       if h not in hits]


def _walk_scope(stmt: ast.AST):
    """Nodes of one scope: does not descend into nested function or class bodies. A ``def``
    statement contributes only what runs in this scope (decorators, defaults): a local name in
    its body is the function's, not this scope's."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield stmt
        args = stmt.args
        roots = [*stmt.decorator_list, *args.defaults, *(d for d in args.kw_defaults if d is not None)]
        for root in roots:
            yield from _walk_scope(root)
        return
    stack = [stmt]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(child)


def _bound_names(targets: Iterable[ast.AST]) -> set[str]:
    names = set()
    for t in targets:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            names |= _bound_names(t.elts)
    return names


_FUNCTIONS: dict[int, tuple[ast.AST, list]] = {}


def _functions(tree: ast.AST) -> list:
    """``(func, cls, parent)`` for every function in the module, nested ones included, parents
    first: ``cls`` names the class of a (non-static) method, ``parent`` the enclosing function
    whose locals a closure sees."""
    cached = _FUNCTIONS.get(id(tree))
    if cached is not None and cached[0] is tree:
        return cached[1]

    def visit(node, cls, parent):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                static = any(_dotted(d) == "staticmethod" for d in child.decorator_list)
                yield child, None if static else cls, parent
                yield from visit(child, None, child)
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, child.name, parent)
            else:
                yield from visit(child, cls, parent)
    found = list(visit(tree, None, None))
    _FUNCTIONS[id(tree)] = (tree, found)
    return found


def _params(func: ast.AST) -> list[str]:
    args = func.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs] + \
        [a.arg for a in (args.vararg, args.kwarg) if a is not None]


def _scope_taints(analysis: _Analysis, tree: ast.AST, module_taint: set[str], calls: bool = False) -> dict:
    """Taint of every function scope, a closure inheriting its enclosing function's."""
    taints: dict = {}
    for func, _cls, parent in _functions(tree):
        taints[func] = analysis.scope_taint(func.body, taints.get(parent, module_taint), calls)
    return taints


def _module_taint(analysis: _Analysis, tree: ast.AST, calls: bool = False) -> set[str]:
    """Module-scope taint: the module body plus core values a function stores into a name it
    declares ``global`` (``def init(): global _M; _M = import_module("run_agent")``)."""
    globals_by_func = {func: names for func, _cls, _parent in _functions(tree)
                       if (names := {n for node in ast.walk(func) if isinstance(node, ast.Global) for n in node.names})}
    taint = analysis.scope_taint(tree.body, set(), calls)
    for _ in range(4 if globals_by_func else 0):
        before = len(taint)
        taints = _scope_taints(analysis, tree, taint, calls)
        for func, names in globals_by_func.items():
            for node in (n for stmt in func.body for n in _walk_scope(stmt)):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None \
                        and analysis.is_core(node.value, taints[func], calls):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    taint |= _bound_names(targets) & names
        if len(taint) == before:
            break
        taint = analysis.scope_taint(tree.body, taint, calls)
    return taint


def _summarise_helpers(analysis: _Analysis, trees: list) -> None:
    """Fixpoint over the whole plugin: which helpers return core values, rebind a parameter, or
    mutate a parameter's private state."""
    nodes_of: dict = {}
    bindings_of: dict = {}
    for _ in range(4):
        before = (len(analysis.core_returners), sum(len(v) for v in analysis.param_patchers.values()),
                  sum(len(v) for v in analysis.param_mutators.values()))
        for _rel, tree in trees:
            taints = _scope_taints(analysis, tree, _module_taint(analysis, tree))
            analysis.module_bindings = analysis.scope_bindings(tree.body)
            for func, cls, _parent in _functions(tree):
                analysis.scope_cls = cls
                if func not in nodes_of:
                    nodes_of[func] = [n for stmt in func.body for n in _walk_scope(stmt)]
                    bindings_of[func] = analysis.scope_bindings(func.body, _params(func))
                nodes = nodes_of[func]
                analysis.bindings = bindings_of[func]
                if any(isinstance(n, ast.Return) and n.value is not None and analysis.is_core(n.value, taints[func])
                       for n in nodes):
                    analysis.core_returners.add(func.name)
                params = [a.arg for a in func.args.posonlyargs + func.args.args]
                # A method is called as ``obj.bind(module, ...)``: its call-site args start after ``self``.
                offset = 1 if cls else 0
                key = (cls, func.name) if cls else func.name
                for index, param in enumerate(params[offset:]):
                    if any(analysis.rebinds(n, {param}, set()) for n in nodes):
                        analysis.param_patchers.setdefault(key, set()).add(index)
                    if any(analysis.private_mutations(n, {param}) for n in nodes):
                        analysis.param_mutators.setdefault(key, set()).add(index)
            analysis.scope_cls = None
        after = (len(analysis.core_returners), sum(len(v) for v in analysis.param_patchers.values()),
                 sum(len(v) for v in analysis.param_mutators.values()))
        if after == before:
            break


def _record_instances(analysis: _Analysis, tree: ast.AST) -> None:
    """What the plugin binds to each ``self.<attr>`` slot (``self._patches = Patches()``)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id == "self":
                    analysis.instances.setdefault("." + target.attr, set()).update(analysis._kinds(node.value))


def core_override_findings(plugin_dir: Path) -> list[str]:
    """``["<target> (<rel>:<line>)", ...]`` for every runtime rebind of Hermes core in the plugin."""
    plugin_dir = Path(plugin_dir)
    parsed: dict[Path, ast.AST] = {}
    for path, rel in _python_files(plugin_dir):
        try:
            parsed[rel] = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except (SyntaxError, ValueError):
            continue
    local = _local_names(plugin_dir)
    runtime = _runtime_files(plugin_dir, parsed, local)
    trees = [(rel, tree) for rel, tree in parsed.items() if rel in runtime]
    analysis = _Analysis(_hermes_top_level(), local)
    for _rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] in local):
                analysis.aliases.update({a.asname: a.name for a in node.names if a.asname})
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                analysis.constants.update({n: node.value.value for n in _bound_names(node.targets)})
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis.class_bases.setdefault(node.name, set()).update(
                    b.id for b in node.bases if isinstance(b, ast.Name))
    for _rel, tree in trees:
        _record_instances(analysis, tree)

    _summarise_helpers(analysis, trees)

    findings: list[str] = []
    for rel, tree in trees:
        module_taint = _module_taint(analysis, tree)
        module_derived = _module_taint(analysis, tree, calls=True)
        taints = _scope_taints(analysis, tree, module_taint)
        derived_taints = _scope_taints(analysis, tree, module_derived, calls=True)
        analysis.module_bindings = analysis.scope_bindings(tree.body)
        scopes = [(tree.body, None, (), module_taint, module_derived)]
        scopes += [(f.body, cls, _params(f), taints[f], derived_taints[f]) for f, cls, _ in _functions(tree)]
        for body, cls, params, taint, derived in scopes:
            analysis.scope_cls = cls
            analysis.bindings = analysis.scope_bindings(body, params)
            for stmt in body:
                for node in _walk_scope(stmt):
                    for target in analysis.rebinds(node, taint, derived):
                        findings.append((rel.as_posix(), getattr(node, "lineno", 0), target))
    _FUNCTIONS.clear()
    return [f"{target} ({rel}:{line})" for rel, line, target in sorted(set(findings))]


def check_core_override(report, plugin_dir: Path) -> None:
    """Fail the report when the plugin's Python rebinds Hermes core at runtime."""
    hits = core_override_findings(plugin_dir)
    report.add(
        "no core override", not hits,
        "rebinds Hermes core at runtime (use a public hook, middleware or provider profile): "
        + "; ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        if hits else "no runtime rebinds of Hermes core",
    )
