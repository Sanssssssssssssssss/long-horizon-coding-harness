from pathlib import Path


def pytest_configure(config):
    # pytest creates basetemp itself, but expects its parent to exist on a fresh clone.
    if config.option.basetemp:
        Path(config.option.basetemp).resolve().parent.mkdir(parents=True, exist_ok=True)
