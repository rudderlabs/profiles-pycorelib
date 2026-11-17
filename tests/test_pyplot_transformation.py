"""Tests for the x/y transformation step of the pyplot model."""
import decimal
import importlib.util
import sys
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def pyplot(sample_model):  # sample_model fixture installs the profiles_rudderstack stubs
    pkg_dir = sys.modules["sample_model"].__file__.rsplit("/", 1)[0]
    name = "pycore_pkg"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, pkg_dir + "/errors.py")
        # pyplot uses a relative import of .errors, so load it inside a package.
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = [pkg_dir]
        sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(
        name + ".pyplot", pkg_dir + "/pyplot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name + ".pyplot"] = module
    spec.loader.exec_module(module)
    return module


class _Material:
    def __init__(self, frames, out):
        self._frames, self._out = frames, out

    def de_ref(self, name, **kwargs):
        return SimpleNamespace(get_df=lambda: self._frames[name].copy())

    def get_output_folder(self):
        return str(self._out)

    def name(self):
        return "plot"


def _run(pyplot, tmp_path, df_x, df_y, x_expr=None, y_expr=None, monkeypatch=None):
    plotted = {}
    monkeypatch.setattr(
        pyplot, "plotGraph",
        lambda x, y, *a, **k: plotted.update(x=x.copy(), y=y.copy()))
    x_axis = {"label": "x", "column": "v", "inputs": ["xin"]}
    y_axis = {"label": "y", "column": "v", "inputs": ["yin"]}
    if x_expr:
        x_axis["transformation"] = x_expr
    if y_expr:
        y_axis["transformation"] = y_expr
    recipe = pyplot.PyPlotRecipe("t", "4x4", False, x_axis, y_axis)
    with warnings.catch_warnings():
        # pandas 2 only warns where pandas 3 raises; treat that warning as a failure.
        warnings.filterwarnings("error", message=".*incompatible dtype.*", category=FutureWarning)
        recipe.execute(_Material({"xin": df_x, "yin": df_y}, tmp_path))
    return plotted


def _ints(*v):
    return pd.DataFrame({"v": pd.Series(v, dtype="int64")})


def test_int_x_fractional(pyplot, tmp_path, monkeypatch):
    p = _run(pyplot, tmp_path, _ints(1, 2, 3), _ints(4, 5, 6), x_expr="x/3", monkeypatch=monkeypatch)
    assert p["x"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert p["y"].tolist() == [4, 5, 6]


def test_int_y_fractional(pyplot, tmp_path, monkeypatch):
    p = _run(pyplot, tmp_path, _ints(1, 2, 3), _ints(1, 2, 3), y_expr="y/1000", monkeypatch=monkeypatch)
    assert p["y"].tolist() == pytest.approx([0.001, 0.002, 0.003])
    assert p["x"].tolist() == [1, 2, 3]


def test_int_both_fractional(pyplot, tmp_path, monkeypatch):
    p = _run(pyplot, tmp_path, _ints(1, 2, 4), _ints(10, 20, 30),
             x_expr="x/4", y_expr="log(y)", monkeypatch=monkeypatch)
    assert p["x"].tolist() == pytest.approx([0.25, 0.5, 1.0])
    assert p["y"].tolist() == pytest.approx([2.302585092994046, 2.995732273553991, 3.4011973816621555])


def test_float_column(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": [1.5, 2.5]})
    p = _run(pyplot, tmp_path, df, df, x_expr="x*2", monkeypatch=monkeypatch)
    assert p["x"].tolist() == [3.0, 5.0]


def test_int_whole_number_result(pyplot, tmp_path, monkeypatch):
    p = _run(pyplot, tmp_path, _ints(1, 2, 3), _ints(1, 2, 3), x_expr="x*10", monkeypatch=monkeypatch)
    assert p["x"].tolist() == [10, 20, 30]
    assert str(p["x"].dtype) == "float64"


def test_nulls(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": [1.0, None, 3.0]})
    p = _run(pyplot, tmp_path, df, df, x_expr="x/2", monkeypatch=monkeypatch)
    assert p["x"].iloc[0] == 0.5 and pd.isna(p["x"].iloc[1]) and p["x"].iloc[2] == 1.5


def test_nullable_int_with_missing_value(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": pd.Series([1, None, 3], dtype="Int64")})
    p = _run(pyplot, tmp_path, df, df, x_expr="x/2", monkeypatch=monkeypatch)
    assert p["x"].iloc[0] == 0.5 and pd.isna(p["x"].iloc[1]) and p["x"].iloc[2] == 1.5


def test_decimal_objects(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": [decimal.Decimal("1"), decimal.Decimal("2")]})
    p = _run(pyplot, tmp_path, df, df, x_expr="x/4", monkeypatch=monkeypatch)
    assert p["x"].tolist() == [0.25, 0.5]


def test_non_numeric_x_without_transformation_untouched(pyplot, tmp_path, monkeypatch):
    dx = pd.DataFrame({"v": ["a", "b"]})
    p = _run(pyplot, tmp_path, dx, _ints(1, 2), y_expr="y/2", monkeypatch=monkeypatch)
    assert p["x"].tolist() == ["a", "b"]
    assert p["y"].tolist() == [0.5, 1.0]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_invalid_expression_raises(pyplot, tmp_path, monkeypatch, axis):
    kwargs = {f"{axis}_expr": "x +* ("}
    with pytest.raises(pyplot.InvalidTransformationError) as exc:
        _run(pyplot, tmp_path, _ints(1), _ints(1), monkeypatch=monkeypatch, **kwargs)
    assert exc.value.__cause__ is not None


def test_non_numeric_with_transformation_keeps_cause_without_values(pyplot, tmp_path, monkeypatch):
    value = "jane@example.com"
    with pytest.raises(pyplot.InvalidTransformationError) as exc:
        _run(pyplot, tmp_path, pd.DataFrame({"v": [value]}), _ints(1), x_expr="x/2", monkeypatch=monkeypatch)
    assert isinstance(exc.value.__cause__, ValueError)
    assert value not in str(exc.value)
    assert type(exc.value.__cause__).__name__ in str(exc.value)
