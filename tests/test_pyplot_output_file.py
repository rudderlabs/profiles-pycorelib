"""Tests for the image file that the pyplot model writes."""
from types import SimpleNamespace

import pandas as pd
import pytest

# pb gives a material name without a file extension, for example this one.
MATERIAL_NAME = "Material_ids_vs_runtime_38fd3a37_39"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class _Material:
    def __init__(self, out, name=MATERIAL_NAME):
        self._out, self._name = out, name

    def de_ref(self, name, **kwargs):
        return SimpleNamespace(get_df=lambda: pd.DataFrame({"v": [1.0, 2.0, 3.0]}))

    def get_output_folder(self):
        return str(self._out)

    def name(self):
        return self._name


def _spec(extension):
    axis = {"label": "v", "column": "v", "inputs": ["in"]}
    return {"title": "t", "size": "4x4", "extension": extension, "x_axis": dict(axis), "y_axis": dict(axis)}


def _execute(pyplot, out, extension, name=MATERIAL_NAME):
    model = pyplot.PyPlotModel(_spec(extension), 99, "0.26.7")
    model.get_material_recipe().execute(_Material(out, name))
    return sorted(p.name for p in out.iterdir())


def test_plot_is_saved_when_material_name_has_no_extension(pyplot, tmp_path):
    assert _execute(pyplot, tmp_path, ".png") == [MATERIAL_NAME + ".png"]
    assert (tmp_path / (MATERIAL_NAME + ".png")).read_bytes().startswith(PNG_SIGNATURE)


def test_extension_of_the_model_sets_the_image_format(pyplot, tmp_path):
    assert _execute(pyplot, tmp_path, ".svg") == [MATERIAL_NAME + ".svg"]
    assert b"<svg" in (tmp_path / (MATERIAL_NAME + ".svg")).read_bytes()


def test_extension_without_leading_dot(pyplot, tmp_path):
    assert _execute(pyplot, tmp_path, "png") == [MATERIAL_NAME + ".png"]


def test_material_name_that_already_has_the_extension_is_kept(pyplot, tmp_path):
    assert _execute(pyplot, tmp_path, ".png", name="plot.PNG") == ["plot.PNG"]


def test_material_name_with_a_dot_still_gets_the_extension(pyplot, tmp_path):
    assert _execute(pyplot, tmp_path, ".png", name="plot.v2") == ["plot.v2.png"]


def test_figure_is_released_after_the_plot_is_saved(pyplot, tmp_path):
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    _execute(pyplot, tmp_path, ".png")
    assert set(plt.get_fignums()) == before


def test_figure_is_released_when_saving_fails(pyplot, tmp_path):
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    recipe = pyplot.PyPlotRecipe("t", "4x4", False, _spec(".png")["x_axis"], _spec(".png")["y_axis"], ".notaformat")
    with pytest.raises(ValueError):
        recipe.execute(_Material(tmp_path))
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("extension, suffix", [(".png", ".png"), ("png", ".png"), (".SVG", ".SVG")])
def test_name_suffix_always_has_one_leading_dot(pyplot, extension, suffix):
    assert pyplot.PyPlotModel(_spec(extension), 99, "0.26.7").get_db_object_name_suffix() == suffix


@pytest.mark.parametrize("extension", [".png", "png", ".svg", ".pdf", ".JPG"])
def test_validate_accepts_file_extensions(pyplot, extension):
    ok, _ = pyplot.PyPlotModel(_spec(extension), 99, "0.26.7").validate()
    assert ok


@pytest.mark.parametrize("extension", [
    "", ".", "..", "../x.png", "/tmp/x.png", ".p/ng", ".png ", "png\n", "a.b", None, 5,
    ".txt", ".xyz",  # plain extensions, but not image formats matplotlib can write
])
def test_validate_rejects_values_that_are_not_a_file_extension(pyplot, extension):
    ok, message = pyplot.PyPlotModel(_spec(extension), 99, "0.26.7").validate()
    assert not ok
    assert "extension" in message


def test_validate_still_rejects_old_schema_versions(pyplot):
    ok, message = pyplot.PyPlotModel(_spec(".png"), 50, "0.26.7").validate()
    assert not ok
    assert "schema version" in message
