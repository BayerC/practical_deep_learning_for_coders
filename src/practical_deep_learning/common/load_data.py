from pathlib import Path

from ddgs import DDGS
from fastai.vision.utils import download_images
from fastcore.all import L


def search_images(keywords: str, max_results: int = 2) -> L:
    return L(DDGS().images(keywords, max_results=max_results)).itemgot("image")


def download_images_to_folder(path: Path, search: str) -> None:
    path.mkdir(parents=True, exist_ok=True)

    download_images(path, urls=search_images(f"{search} photo"))
