from pathlib import Path
from tempfile import TemporaryDirectory

from src.practical_deep_learning.common.load_data import (
    download_images_to_folder,
    get_existing_images,
)


def test_download_images():
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "data" / "birds"

        download_images_to_folder(test_path, "cat", 10)
        assert test_path.exists(), "Directory should exist"
        assert test_path.is_dir(), "Path should be a directory"

        # Check for common image formats
        image_files = get_existing_images(test_path)
        assert len(image_files) == 10, "Should have 10 images"
