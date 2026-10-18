"""Pure function dependency projection for experimental Skill learners.

Ported from the frozen extract/extract_v2 research implementation. Callers own
teacher provenance, the original validator, bank storage and semantic probes.
This conservative AST scope filter is not a general Python sandbox.
"""
from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from hashlib import sha256

Validator = Callable[[bytes, Sequence[str]], str | None]
_FORBIDDEN_NAMES = {
    'open', '__import__', 'eval', 'exec', 'compile', 'input', 'globals', 'locals',
    'getattr', 'setattr', 'delattr', 'vars', 'breakpoint', '__builtins__',
}
_IO_METHODS = {
    'read', 'write', 'read_text', 'write_text', 'read_bytes', 'write_bytes', 'open',
    'load', 'dump', 'system', 'popen', 'unlink', 'remove', 'rename', 'mkdir',
}


def _bindings(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Import):
        return {item.asname or item.name.split('.')[0] for item in node.names}
    if isinstance(node, ast.ImportFrom):
        return {item.asname or item.name for item in node.names}
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _check_scope(tree: ast.Module) -> None:
    for top in tree.body:
        for node in ast.walk(top):
            if isinstance(node, ast.ClassDef):
                raise ValueError('Class construction is outside the pure function helper scope')
            if isinstance(node, ast.ImportFrom) and any(item.name == '*' for item in node.names):
                raise ValueError('Star imports lack explicit static bindings')
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node is not top:
                raise ValueError('Nested import is not a static module dependency')
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _FORBIDDEN_NAMES:
                raise ValueError('Impure or dynamic reference: ' + node.id)
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                raise ValueError('Dynamic dunder access is outside the helper scope')
            if isinstance(node, ast.Call):
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise ValueError('Indirect call requires a proven dependency contract')
                if isinstance(node.func, ast.Name) and node.func.id == 'type':
                    raise ValueError('Dynamic type construction is outside the helper scope')
                if isinstance(node.func, ast.Attribute) and node.func.attr in _IO_METHODS:
                    raise ValueError('File or external IO call: ' + node.func.attr)


def compile_helper(
    source: bytes,
    identifiers: Sequence[str],
    *,
    roots: Sequence[str],
    validator: Validator,
) -> tuple[str, dict]:
    """Project declared roots and dependencies, then apply the caller's original guard.

    The validator may normalize source, as the existing helper guard does; both
    sides of that boundary retain the frozen pure-function scope restrictions.
    Unknown dependencies remain the validator's obligation. No code is executed.
    """
    if not roots or isinstance(roots, str) or any(not isinstance(root, str) or not root for root in roots):
        raise ValueError('roots must contain explicit nonempty binding names')
    if not callable(validator):
        raise TypeError('an explicit original validator is required')
    tree = ast.parse(source.decode('utf-8'))
    declarations = {}
    for node in tree.body:
        for name in _bindings(node):
            if name in declarations:
                raise ValueError('Ambiguous module binding: ' + name)
            declarations[name] = node
    pending, selected = list(roots), set()
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in declarations:
            raise ValueError('Missing required entrypoint: ' + name)
        selected.add(name)
        pending.extend(node.id for node in ast.walk(declarations[name])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id in declarations and node.id not in selected)
    retained = ast.Module(body=[node for node in tree.body if _bindings(node) & selected
        or isinstance(node, ast.ImportFrom) and node.module == '__future__'], type_ignores=[])
    _check_scope(retained)
    candidate = ast.unparse(retained) + '\n'
    guarded = validator(candidate.encode('utf-8'), identifiers)
    if guarded is None:
        raise ValueError('Original validator rejected the dependency closure')
    validated = ast.parse(guarded)
    _check_scope(validated)
    exports = {name for node in validated.body for name in _bindings(node)}
    if set(roots) - exports:
        raise ValueError('Original validator dropped declared roots')
    compile(guarded, 'learned_parser.py', 'exec')
    return guarded, {
        'source_sha256': sha256(source).hexdigest(), 'helper_sha256': sha256(guarded.encode('utf-8')).hexdigest(),
        'roots': list(roots), 'selected_bindings': sorted(selected),
        'excluded_bindings': sorted(set(declarations) - selected),
        'source_mutated': False, 'legacy_guard_relaxed': False,
        'pure_function_scope': True, 'static_import_bindings_only': True,
        'no_classes_or_indirect_calls': True,
    }
