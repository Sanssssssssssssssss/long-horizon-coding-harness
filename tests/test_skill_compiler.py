import ast
import builtins
import importlib.util
import symtable

import pytest

from benchmarks.skill_compiler import compile_helper


def dependency_validator(source, identifiers):
    """Independent caller guard for the compiler's unresolved-name boundary."""
    text = source.decode('utf-8')
    if any(identifier in text for identifier in identifiers):
        return None
    table = symtable.symtable(text, 'helper.py', 'exec')
    bound = {symbol.get_name() for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()} | set(dir(builtins))
    def unresolved(scope):
        return any(s.is_global() and s.is_referenced() and s.get_name() not in bound for s in scope.get_symbols()) or any(unresolved(child) for child in scope.get_children())
    return None if unresolved(table) else text


SOURCE = b'''from __future__ import annotations
import re
import os
PREFIX = "value:"
PATTERN = re.compile(r"[a-z]+")
def normalize(text):
    return PREFIX + PATTERN.search(text).group()
def parse_script(text):
    return {"text": normalize(text)}
def to_dot(graph):
    return graph["text"]
def main():
    return open("unrelated-cli-file").read()
if __name__ == "__main__":
    main()
'''


def test_dependency_closure_preserves_behavior_and_survives_save_reload(tmp_path):
    helper, metadata = compile_helper(SOURCE, (), roots=('parse_script', 'to_dot'), validator=dependency_validator)
    assert set(metadata['selected_bindings']) == {'parse_script', 'to_dot', 'normalize', 'PREFIX', 'PATTERN', 're'}
    assert {'main', 'os'} <= set(metadata['excluded_bindings'])
    assert not any(isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names) for n in ast.parse(helper).body)
    path = tmp_path / 'learned_parser.py'; path.write_bytes(helper.encode('utf-8'))
    spec = importlib.util.spec_from_file_location('saved_compiler_helper', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.to_dot(module.parse_script('  abc  ')) == 'value:abc'
    assert path.read_bytes() == helper.encode('utf-8') and SOURCE.endswith(b'    main()\n')


@pytest.mark.parametrize('body', [
    'def parse_script(text):\n    return missing_dependency(text)\n',
    'def parse_script(text):\n    return __import__("os").getcwd()\n',
    'def parse_script(text):\n    return open("/app/script.txt").read()\n',
    'def parse_script(text):\n    return exec(text)\n',
    'def parse_script(text):\n    return eval(text)\n',
    'class Graph:\n    pass\ndef parse_script(text):\n    return Graph()\n',
    'def parse_script(text):\n    return type("Graph", (), {})\n',
    'def parse_script(text):\n    return ().__class__.__base__.__subclasses__()\n',
])
def test_existing_unsafe_or_incomplete_cases_stay_rejected(body):
    source = (body + '\ndef to_dot(graph):\n    return "digraph g {}"\n').encode('utf-8')
    with pytest.raises(ValueError):
        compile_helper(source, (), roots=('parse_script', 'to_dot'), validator=dependency_validator)


def test_validator_cannot_replace_safe_projection_with_dynamic_code():
    with pytest.raises(ValueError, match='exec'):
        compile_helper(SOURCE, (), roots=('parse_script', 'to_dot'), validator=lambda source, ids: 'exec("pass")\n')


def test_validator_cannot_drop_declared_exports():
    with pytest.raises(ValueError, match='dropped declared roots'):
        compile_helper(SOURCE, (), roots=('parse_script', 'to_dot'), validator=lambda source, ids: 'pass\n')


def test_explicit_roots_and_original_validator_are_required():
    with pytest.raises(ValueError, match='Missing required'):
        compile_helper(SOURCE, (), roots=('unknown',), validator=dependency_validator)
    with pytest.raises(TypeError, match='validator'):
        compile_helper(SOURCE, (), roots=('parse_script',), validator=None)
