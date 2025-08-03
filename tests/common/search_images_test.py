from src.practical_deep_learning.common.search_images import search_images


def test_search_images():
    result = search_images("test", max_results=5)
    assert isinstance(result, list)
    assert len(result) <= 5
