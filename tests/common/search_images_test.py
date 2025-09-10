from unittest.mock import patch

from src.practical_deep_learning.common.search_images import search_images


def test_search_images():
    # Mock the DDGS response to avoid external API calls during tests
    mock_results = [
        {"image": "https://example.com/image1.jpg", "title": "Test Image 1"},
        {"image": "https://example.com/image2.jpg", "title": "Test Image 2"},
        {"image": "https://example.com/image3.jpg", "title": "Test Image 3"},
    ]

    with patch("src.practical_deep_learning.common.search_images.DDGS") as mock_ddgs:
        mock_ddgs.return_value.images.return_value = mock_results
        result = search_images("test", max_results=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        assert len(result) == 3  # Should return the mocked results
