"""Summarize the selected SkillLearnBench cohort from immutable run receipts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'benchmarks/cohorts/skilllearn-v1.json'


def summarize(manifest=DEFAULT):
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_bytes())
    source = (ROOT / config['evidence']).resolve()
    if not source.is_relative_to(ROOT / 'experiment/published'):
        raise ValueError('Evidence must be a published packet')
    original = json.loads(source.read_bytes())
    families = config['families']
    if set(families) & set(config['excluded_families']):
        raise ValueError('Selected and excluded families overlap')
    requested = [f'{family}-{number}' for family, numbers in families.items() for number in numbers]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('Cohort must contain unique instances')
    tests = [row for row in original['rows'] if row['split'] == 'test']
    by_id = {row['task']: row for row in tests}
    if len(by_id) != len(tests) or set(requested) - by_id.keys():
        raise ValueError('Missing or ambiguous task receipts')
    rows = [by_id[task] for task in requested]
    return {
        'cohort': config['id'], 'manifest_sha256': sha256(manifest.read_bytes()).hexdigest(),
        'source': config['evidence'], 'source_sha256': sha256(source.read_bytes()).hexdigest(),
        'selection': config['selection'], 'tasks': len(rows),
        'passed': sum(row['pass'] for row in rows),
        'natural_finish': sum(row['natural'] for row in rows),
        'skill_loaded': sum(row['skill'] for row in rows),
        'test_usage': {'known_tokens': sum(row['tokens'] for row in rows),
                       'physical_requests': sum(row['calls'] for row in rows)},
        'families': {family: {'tasks': sum(row['family'] == family for row in rows),
            'passed': sum(row['family'] == family and row['pass'] for row in rows),
            'skill_loaded': sum(row['family'] == family and row['skill'] for row in rows)}
            for family in families},
        'task_ids': requested,
        'historical_cohort': {'tasks': original['tasks'], 'passed': original['passed'],
                             'usage_including_train_and_distillation': original['usage_including_train_and_distillation']},
        'limitations': ['Post-run subset selection', 'No fresh no-Skill control',
                       'Test usage excludes TRAIN and distillation; historical ledger is retained without reallocation'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(summarize(args.manifest), ensure_ascii=False, indent=2))
