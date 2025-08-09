from pathlib import Path
from tempfile import TemporaryDirectory

from src.practical_deep_learning.common.load_data import download_images


def test_download_images():
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "data" / "birds"

        download_images(test_path)
        assert test_path.exists(), "Directory should exist"
        assert test_path.is_dir(), "Path should be a directory"
