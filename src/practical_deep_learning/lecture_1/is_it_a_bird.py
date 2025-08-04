import time
from pathlib import Path

from ddgs import DDGS
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    RandomSplitter,
    Resize,
    parent_label,
)
from fastai.vision.utils import (
    download_images,
    get_image_files,
    resize_images,
    verify_images,
)
from fastcore.all import L


def search_images(keywords: str, max_results: int = 200) -> L:
    return L(DDGS().images(keywords, max_results=max_results)).itemgot("image")


def load_data() -> Path:
    data_path = Path(__file__).parent / "data" / "bird_or_not"

    searches = "forest", "bird"
    for search in searches:
        dest = data_path / search
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f"{search} photo"))
        time.sleep(5)
        resize_images(data_path / search, max_size=400, dest=data_path / search)

    failed = verify_images(get_image_files(data_path))
    failed.map(Path.unlink)
    return data_path


def is_it_a_bird() -> None:
    data_path = load_data()
    print(f"Data loaded from path: {data_path}")

    dls = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method="squish")],
    ).dataloaders(data_path, bs=32)

    dls.show_batch(max_n=6)


if __name__ == "__main__":
    is_it_a_bird()
