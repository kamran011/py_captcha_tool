"""The generated datasets must be reproducible: same seed, same bytes (the labels were never the problem,
the pixels were: the captcha library draws its randomness from `secrets`, which cannot be seeded)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import ml.generate_dataset as gen


def digest(folder: Path) -> list[str]:
    return [hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.png"))]


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "OUT", tmp_path)
    return tmp_path


def test_same_seed_gives_byte_identical_images_and_labels(out):
    gen.generate_split("a", 12, seed=3)
    gen.generate_split("b", 12, seed=3)
    assert digest(out / "a") == digest(out / "b") and len(digest(out / "a")) == 12
    assert (out / "a" / "labels.csv").read_text() == (out / "b" / "labels.csv").read_text()


def test_different_seeds_give_different_images(out):
    gen.generate_split("a", 12, seed=3)
    gen.generate_split("b", 12, seed=4)
    assert digest(out / "a") != digest(out / "b")


def test_generation_does_not_depend_on_other_randomness_in_the_process(out):
    import random
    import secrets

    gen.generate_split("a", 8, seed=3)
    random.seed(999)
    _ = [secrets.randbelow(100) for _ in range(50)]            # consume OS and global randomness
    gen.generate_split("b", 8, seed=3)
    assert digest(out / "a") == digest(out / "b")


def test_math_dataset_is_reproducible(out):
    gen.generate_math("m1", 12, seed=4)
    gen.generate_math("m2", 12, seed=4)
    assert digest(out / "m1") == digest(out / "m2")
    assert (out / "m1" / "labels.csv").read_text() == (out / "m2" / "labels.csv").read_text()


def test_the_seeded_stand_in_matches_the_secrets_api_the_library_uses():
    import random

    s = gen._SeededSecrets(random.Random(1))
    assert 0 <= s.randbelow(10) < 10 and 0 <= s.randbits(32) < 2**32 and s.choice([1, 2, 3]) in (1, 2, 3)
