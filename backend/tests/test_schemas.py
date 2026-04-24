import pytest
from pydantic import ValidationError

from models.schemas import CategoriesConfig, CategoryDef


def test_minimal_valid_category() -> None:
    c = CategoryDef(name="Food", prompts=["a photo of food"])
    assert c.name == "Food"
    assert c.prompts == ["a photo of food"]


def test_category_strips_empty_prompts() -> None:
    c = CategoryDef(name="X", prompts=["  valid  ", "", "   ", "also valid"])
    assert c.prompts == ["valid", "also valid"]


def test_category_rejects_all_empty_prompts() -> None:
    with pytest.raises(ValidationError):
        CategoryDef(name="X", prompts=["", "   "])


def test_category_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        CategoryDef(name="", prompts=["ok"])


def test_category_rejects_too_many_prompts() -> None:
    with pytest.raises(ValidationError):
        CategoryDef(name="X", prompts=["p"] * 21)


def test_full_config_defaults() -> None:
    cfg = CategoriesConfig(categories=[{"name": "A", "prompts": ["a"]}])
    assert cfg.threshold == 0.22
    assert cfg.batch_size == 32


def test_threshold_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CategoriesConfig(categories=[{"name": "A", "prompts": ["a"]}], threshold=1.5)
    with pytest.raises(ValidationError):
        CategoriesConfig(categories=[{"name": "A", "prompts": ["a"]}], threshold=-0.1)


def test_batch_size_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CategoriesConfig(categories=[{"name": "A", "prompts": ["a"]}], batch_size=0)
    with pytest.raises(ValidationError):
        CategoriesConfig(categories=[{"name": "A", "prompts": ["a"]}], batch_size=1000)


def test_config_requires_at_least_one_category() -> None:
    with pytest.raises(ValidationError):
        CategoriesConfig(categories=[])
