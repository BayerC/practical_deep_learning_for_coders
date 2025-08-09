from typing import Any

from ddgs import DDGS


def search_images(keywords: str, max_results: int = 200) -> list[dict[str, Any]]:
    # test
    return DDGS().images(keywords, max_results=max_results)
