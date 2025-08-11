from pathlib import Path

from ddgs import DDGS
from fastai.vision.utils import (
    download_images,
)


def get_existing_images(path: Path) -> list[Path]:
    """Get all existing image files in the given path."""
    return (
        list(path.glob("*.jpg"))
        + list(path.glob("*.jpeg"))
        + list(path.glob("*.png"))
        + list(path.glob("*.gif"))
        + list(path.glob("*.bmp"))
    )


def search_images(keywords: str, max_results: int) -> list[str]:
    return [item["image"] for item in DDGS().images(keywords, max_results=max_results)]


def download_images_to_folder(path: Path, search: str, num_images: int) -> None:
    path.mkdir(parents=True, exist_ok=True)

    # Check for existing images
    existing_images = get_existing_images(path)

    if len(existing_images) >= num_images:
        print(f"Already have {len(existing_images)} images, skipping download")
        return

    # Calculate how many more images we need
    images_needed = num_images - len(existing_images)
    print(
        f"Found {len(existing_images)} existing images, downloading {images_needed} more"
    )

    urls = search_images(f"{search} photo", images_needed)
    download_images(path, urls=urls)
