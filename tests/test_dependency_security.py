from __future__ import annotations

import io
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from parsemedicalexams.utils import preprocess_page_image

ROOT = Path(__file__).resolve().parents[1]


def test_patched_dependency_versions_are_installed() -> None:
    assert tuple(map(int, version("pillow").split("."))) >= (13, 0, 0)
    assert tuple(map(int, version("certifi").split("."))) >= (2026, 10, 19)
    assert tuple(map(int, version("filelock").split("."))) >= (4, 2, 1)
    assert tuple(map(int, version("msgpack").split("."))) >= (1, 2, 3)
    assert tuple(map(int, version("python-dotenv").split("."))) >= (1, 2, 2)
    assert tuple(map(int, version("pytest").split("."))) >= (9, 0, 3)
    assert tuple(map(int, version("pygments").split("."))) >= (2, 20, 0)


def test_pillow_accepts_valid_image_and_rejects_malformed_input() -> None:
    buffer = io.BytesIO()
    Image.new("L", (2, 2), color=128).save(buffer, format="PNG")

    with Image.open(io.BytesIO(buffer.getvalue())) as image:
        image.load()
        assert image.size == (2, 2)

    for malformed in (b"not a medical image", buffer.getvalue()[:40]):
        with pytest.raises((UnidentifiedImageError, OSError)):
            with Image.open(io.BytesIO(malformed)) as image:
                image.load()


@pytest.mark.parametrize("size", [(2000, 1000), (1000, 2000), (100, 50)])
def test_pillow_preserves_page_preprocessing(size: tuple[int, int]) -> None:
    with Image.new("RGB", size, color=(128, 128, 128)) as image:
        image.paste((64, 64, 64), (0, 0, size[0] // 2, size[1]))
        processed = preprocess_page_image(image)

    ratio = min(1.0, 1000 / max(size))
    assert processed.mode == "L"
    assert processed.size == (int(size[0] * ratio), int(size[1] * ratio))
    assert processed.getpixel((0, 0)) < 64
    assert processed.getpixel((processed.width - 1, 0)) > 128

    output = io.BytesIO()
    processed.save(output, format="JPEG", quality=80)
    with Image.open(io.BytesIO(output.getvalue())) as reloaded:
        reloaded.load()
        assert reloaded.mode == "L"
        assert reloaded.size == processed.size


def test_dependency_manifests_use_registry_sources_only() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        manifest = tomllib.load(file)
    with (ROOT / "uv.lock").open("rb") as file:
        lock = tomllib.load(file)

    declared = list(manifest["project"]["dependencies"])
    for dependencies in manifest["project"].get("optional-dependencies", {}).values():
        declared.extend(dependencies)
    assert not any(
        dependency.lower().startswith(("file:", "git:", "git+", "http:", "https:"))
        for dependency in declared
    )

    for package in lock["package"]:
        source = package.get("source", {})
        if source.get("editable") == ".":
            continue
        assert source == {"registry": "https://pypi.org/simple"}
