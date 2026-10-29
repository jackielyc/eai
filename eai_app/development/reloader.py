"""Conservative, transactional function-body updates for a running application.

No module initializers, imports, decorators or constructors are rerun. Functions
keep their identities, globals and closures, including Qt's connected callbacks.
Structural changes are reported as requiring a restart rather than guessed at.
"""
from __future__ import annotations
import ast
import __future__
import copy
from collections import Counter
import sys
import types
from pathlib import Path


class RestartRequired(ValueError):
    pass


def _shape(tree):
    tree = copy.deepcopy(tree)
    # Only bodies of top-level functions and direct class methods may change.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Pass()]
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    member.body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def _functions(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield None, node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node.name, member


def _target(module, cls, node):
    owner = vars(module) if cls is None else vars(vars(module)[cls])
    value = owner.get(node.name)
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    elif isinstance(value, property):
        decorators = [ast.unparse(d) for d in node.decorator_list]
        if any(d.endswith('.setter') for d in decorators): value = value.fset
        elif any(d.endswith('.deleter') for d in decorators): value = value.fdel
        else: value = value.fget
    if not isinstance(value, types.FunctionType):
        raise RestartRequired(f'{cls or module.__name__}.{node.name}: decorated/native function')
    if value.__module__ != module.__name__:
        raise RestartRequired(f'{node.name}: function belongs to another module')
    return value


def prepare(module, old_source, new_source, filename):
    old, new = ast.parse(old_source), ast.parse(new_source)
    if _shape(old) != _shape(new):
        raise RestartRequired('imports, constants, signatures, decorators or class structure changed')
    def identity(cls, node):
        accessors = tuple(ast.unparse(d) for d in node.decorator_list
                          if ast.unparse(d).endswith(('.setter', '.deleter')))
        return cls, node.name, accessors
    counts = Counter(identity(cls, node) for cls, node in _functions(new))
    edits = []
    for (cls, before), (_, after) in zip(_functions(old), _functions(new)):
        if ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False):
            continue
        if after.name in ('__init__', '__new__', '__del__', 'build') or after.name.startswith(('build_', '_build_')):
            raise RestartRequired(f'{after.name}: construction/layout change needs restart')
        if counts[identity(cls, after)] > 1:
            raise RestartRequired(f'{after.name}: duplicate definition requires restart')
        target = _target(module, cls, after)
        fn = copy.deepcopy(after)
        fn.decorator_list = []
        fn.returns = None
        for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
            arg.annotation = None
        for arg in (fn.args.vararg, fn.args.kwarg):
            if arg: arg.annotation = None
        fn.args.defaults = [ast.Constant(None) for _ in fn.args.defaults]
        fn.args.kw_defaults = [ast.Constant(None) if x is not None else None for x in fn.args.kw_defaults]
        body = [fn]
        if cls:
            body = [ast.ClassDef(name='_ReloadStage', bases=[], keywords=[], body=body, decorator_list=[])]
        tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        namespace = {'__name__': module.__name__}
        future_flags = sum(getattr(__future__, name).compiler_flag for name in __future__.all_feature_names)
        exec(compile(tree, filename, 'exec', flags=target.__code__.co_flags & future_flags,
                     dont_inherit=True), namespace)
        staged = vars(namespace['_ReloadStage'])[fn.name] if cls else namespace[fn.name]
        if staged.__code__.co_freevars != target.__code__.co_freevars:
            raise RestartRequired(f'{fn.name}: closure layout changed')
        if staged.__code__.co_flags != target.__code__.co_flags:
            raise RestartRequired(f'{fn.name}: function execution mode changed')
        edits.append((target, staged.__code__))
    return edits


class SourceReloader:
    def __init__(self, root, busy=lambda: '', modules=None):
        self.root = Path(root).resolve()
        self.busy = busy
        self.modules = sys.modules if modules is None else modules
        self.sources = self.read_sources()
        self.last_message = '代码监视已就绪'

    def read_sources(self):
        return {p: p.read_text(encoding='utf-8') for p in self.root.rglob('*.py')}

    def changed_paths(self):
        current = self.read_sources()
        return sorted(p for p in self.sources.keys() | current.keys() if self.sources.get(p) != current.get(p))

    def check(self):
        current = self.read_sources()
        changed = [p for p in self.sources.keys() | current.keys() if self.sources.get(p) != current.get(p)]
        if not changed:
            return 'unchanged', '代码无变化'
        reason = self.busy()
        if reason:
            return 'deferred', '等待空闲后更新：' + reason
        edits = []
        try:
            by_path = {}
            for module in tuple(self.modules.values()):
                path = getattr(module, '__file__', None)
                if path:
                    by_path.setdefault(Path(path).resolve(), module)
            for path in sorted(changed):
                if path not in current or path not in self.sources:
                    raise RestartRequired(f'{path.name}: module added or removed')
                compile(current[path], str(path), 'exec')  # Validate even unloaded workers.
                if path not in by_path:
                    # A fresh worker/import will read the changed source. Remove its
                    # timestamp-based cache so same-second edits cannot be missed.
                    import importlib.util
                    Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
                    continue
                rel = path.relative_to(self.root).as_posix()
                if rel.startswith(('development/', 'adapters/ros/', 'platform/')) or rel in (
                    'application/entrypoint.py', 'application/execution_client.py',
                    'adapters/robot/execution_service.py'):
                    raise RestartRequired(f'{rel}: process/ROS lifecycle change')
                edits.extend(prepare(by_path[path], self.sources[path], current[path], str(path)))
        except (SyntaxError, ValueError, KeyError, OSError) as exc:
            return 'restart' if isinstance(exc, RestartRequired) else 'error', str(exc)
        rollback = [(fn, fn.__code__) for fn, _ in edits]
        try:
            for fn, code in edits:
                fn.__code__ = code
        except Exception as exc:
            for fn, code in rollback:
                fn.__code__ = code
            return 'error', str(exc)
        self.sources = current
        return 'applied', f'已更新 {len(changed)} 个文件、{len(edits)} 个方法；未加载模块在下次导入/启动时生效'
