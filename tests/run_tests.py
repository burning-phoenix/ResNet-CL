import pytest
import sys
from pathlib import Path
import collections
import collections.abc
import sys

project_root = Path(__file__).resolve().parent.parent # I dont know why we need this used LLM to make sure test run properly without this it fails I dont know why
sys.path.insert(0, str(project_root))


if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable # same here


if __name__ == "__main__":
    tests_dir = Path(__file__).resolve().parent
    default_args = [
        str(tests_dir / "test_metric.py"),
        str(tests_dir / "test_train.py"),
        "-k", "test_e_",
        "-v",
    ]
    exit_code = pytest.main(sys.argv[1:] or default_args)
    sys.exit(exit_code)
