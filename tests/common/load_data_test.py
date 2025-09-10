from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.practical_deep_learning.common.load_data import (
    download_images_to_folder,
    get_existing_images,
)


def test_download_images():
    with TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "data" / "birds"

        # Mock the DDGS response to avoid external API calls during tests
        mock_urls = [
            "https://example.com/cat1.jpg",
            "https://example.com/cat2.jpg",
            "https://example.com/cat3.jpg",
            "https://example.com/cat4.jpg",
            "https://example.com/cat5.jpg",
            "https://example.com/cat6.jpg",
            "https://example.com/cat7.jpg",
            "https://example.com/cat8.jpg",
            "https://example.com/cat9.jpg",
            "https://example.com/cat10.jpg",
        ]

        with patch(
            "src.practical_deep_learning.common.load_data.search_images"
        ) as mock_search:
            mock_search.return_value = mock_urls
            # Mock the download_images function to create dummy files instead of downloading
            with patch(
                "src.practical_deep_learning.common.load_data.download_images"
            ) as mock_download:
                mock_download.side_effect = lambda path, urls: [
                    (path / f"image_{i}.jpg").touch() for i in range(len(urls))
                ]

                download_images_to_folder(test_path, "cat", 10)
                assert test_path.exists(), "Directory should exist"
                assert test_path.is_dir(), "Path should be a directory"

                # Check for common image formats
                image_files = get_existing_images(test_path)
                assert len(image_files) == 10, "Should have 10 images"
