"""New domain with independent checks, not present in the extraction trajectory."""
CORE = '''def decode_cell(value, *, offset=0, as_int=False):
    if value == "CRASH":
        raise RuntimeError("infrastructure failed")
    number = float(value) + offset
    return int(number) if as_int else number
'''
TASK = '''Return only JSON {"code":"complete adapter.py module source"}.
Add batch adapters to an existing measurement decoder. Import the unchanged core function
using from core import decode_cell. Its source is below. Do not redefine it.
Implement decode_many(values, *, on_error='raise', **core_options), returning a list.
Allowed on_error policies are raise, skip and null. Validate the policy before consuming
values, including empty inputs. Pass only core_options to decode_cell. Catch only
TypeError and ValueError from decode_cell: raise propagates, skip omits that element,
null inserts None. Other exceptions propagate. Iterable inputs may be single-use.
Implement decode_records(records, *, field='value', destination='decoded',
on_error='raise', **core_options), returning new dictionaries. Deep-copy each accepted
input record, preserve every unrelated field, and set destination to the decoded result.
Apply the same policies to missing fields (a ValueError) and decoder errors. Null retains
the copied record with destination=None; skip omits it. Inputs, including nested values,
must remain independent of the returned records. Validate policy even for empty records.
'''
GOLD = '''from copy import deepcopy
from core import decode_cell

def _policy(value):
    if value not in ("raise", "skip", "null"):
        raise ValueError("invalid on_error")

def decode_many(values, *, on_error="raise", **core_options):
    _policy(on_error)
    out = []
    for value in values:
        try:
            decoded = decode_cell(value, **core_options)
        except (TypeError, ValueError):
            if on_error == "raise":
                raise
            if on_error == "skip":
                continue
            decoded = None
        out.append(decoded)
    return out

def decode_records(records, *, field="value", destination="decoded", on_error="raise", **core_options):
    _policy(on_error)
    out = []
    for record in records:
        try:
            if field not in record:
                raise ValueError("missing field")
            decoded = decode_cell(record[field], **core_options)
        except (TypeError, ValueError):
            if on_error == "raise":
                raise
            if on_error == "skip":
                continue
            decoded = None
        result = deepcopy(record)
        result[destination] = decoded
        out.append(result)
    return out
'''
CHECKS = '''from copy import deepcopy
import pytest
from adapter import decode_many, decode_records

@pytest.mark.parametrize("policy,expected", [("skip", [2.0, 3.0]), ("null", [2.0, None, 3.0])])
def test_scalar_policies(policy, expected):
    assert decode_many(iter(["2", "bad", "3"]), on_error=policy) == expected

def test_scalar_default_error():
    with pytest.raises(ValueError):
        decode_many(["bad"])

def test_core_options():
    assert decode_many(["2.8"], offset=1, as_int=True) == [3]

@pytest.mark.parametrize("function", [decode_many, decode_records])
@pytest.mark.parametrize("policy", ["unknown", False, None])
def test_empty_policy_validation(function, policy):
    with pytest.raises(ValueError):
        function([], on_error=policy)

@pytest.mark.parametrize("policy", ["skip", "null"])
def test_unexpected_errors_propagate(policy):
    with pytest.raises(RuntimeError):
        decode_many(["CRASH"], on_error=policy)

def test_record_options_and_copy():
    records = [{"reading": "2.8", "extra": {"labels": ["x"]}}]
    before = deepcopy(records)
    out = decode_records(records, field="reading", destination="amount", offset=1, as_int=True)
    assert out == [{"reading": "2.8", "amount": 3, "extra": {"labels": ["x"]}}]
    assert records == before
    out[0]["extra"]["labels"].append("changed")
    assert records == before

@pytest.mark.parametrize("policy,expected", [("skip", [{"value": "3", "decoded": 3.0}]),
    ("null", [{"other": 1, "decoded": None}, {"value": "bad", "decoded": None}, {"value": "3", "decoded": 3.0}])])
def test_record_missing_and_bad(policy, expected):
    assert decode_records(iter([{"other": 1}, {"value": "bad"}, {"value": "3"}]), on_error=policy) == expected

def test_record_missing_default_error():
    with pytest.raises(ValueError):
        decode_records([{}])

def test_record_unexpected_error():
    with pytest.raises(RuntimeError):
        decode_records([{"value": "CRASH"}], on_error="null")
'''
