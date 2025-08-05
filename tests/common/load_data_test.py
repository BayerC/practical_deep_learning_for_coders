from pathlib import Path
from tempfile import TemporaryDirectory

from src.practical_deep_learning.common.load_data import load_data


def test_load_data():
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir)
        load_data(test_path)
        assert not test_path.exists(), (
            f"Expected {test_path} to not exist after load_data call."
        )
