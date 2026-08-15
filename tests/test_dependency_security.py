from __future__ import annotations

import io
from importlib.metadata import version
from pathlib import Path

import pytest
import tomllib
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]


def test_patched_dependency_versions_are_installed() -> None:
    assert tuple(map(int, version("pillow").split("."))) >= (12, 3, 0)
    assert tuple(map(int, version("python-dotenv").split("."))) >= (1, 2, 2)
    assert tuple(map(int, version("pytest").split("."))) >= (9, 0, 3)
    assert tuple(map(int, version("pygments").split("."))) >= (2, 20, 0)


def test_pillow_accepts_valid_image_and_rejects_malformed_input() -> None:
    buffer = io.BytesIO()
    Image.new("L", (2, 2), color=128).save(buffer, format="PNG")

    with Image.open(io.BytesIO(buffer.getvalue())) as image:
        image.load()
        assert image.size == (2, 2)

    with pytest.raises((UnidentifiedImageError, OSError)):
        with Image.open(io.BytesIO(b"not a medical image")) as image:
            image.load()


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
