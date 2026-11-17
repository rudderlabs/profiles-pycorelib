"""Tests for the x/y transformation step of the pyplot model."""
import decimal
import traceback
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest


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


def _failure(pyplot, tmp_path, monkeypatch, axis, frame, expr="{v}/2"):
    """Run a transformation that must fail; return the error and its full formatted traceback."""
    other = _ints(1)
    frames = (frame, other) if axis == "x" else (other, frame)
    with pytest.raises(pyplot.InvalidTransformationError) as exc:
        _run(pyplot, tmp_path, *frames, monkeypatch=monkeypatch, **{f"{axis}_expr": expr.format(v=axis)})
    return exc.value, "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize("axis", ["x", "y"])
def test_text_column_error_has_no_cell_value_anywhere_in_traceback(pyplot, tmp_path, monkeypatch, axis):
    value = "jane@example.com"
    error, formatted = _failure(pyplot, tmp_path, monkeypatch, axis, pd.DataFrame({"v": [value]}))
    assert value not in str(error)
    assert value not in formatted
    assert error.__cause__ is None and error.__suppress_context__


@pytest.mark.parametrize("axis", ["x", "y"])
def test_zip_code_text_column_still_fails(pyplot, tmp_path, monkeypatch, axis):
    frame = pd.DataFrame({"v": ["02139", "10001"]})
    error, formatted = _failure(pyplot, tmp_path, monkeypatch, axis, frame, expr="{v}*1")
    assert "02139" not in formatted and "10001" not in formatted
    assert f"{axis.upper()}-axis column 'v' (" in str(error)
    assert str(frame["v"].dtype) in str(error)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_non_numeric_column_is_not_reported_as_invalid_expression(pyplot, tmp_path, monkeypatch, axis):
    error, _ = _failure(pyplot, tmp_path, monkeypatch, axis, pd.DataFrame({"v": ["a"]}))
    assert "cannot be converted to numbers" in str(error)
    assert "invalid transformation-expression" not in str(error)


def test_invalid_expression_is_not_reported_as_a_column_problem(pyplot, tmp_path, monkeypatch):
    error, _ = _failure(pyplot, tmp_path, monkeypatch, "x", _ints(1), expr="x +* (")
    assert "invalid transformation-expression" in str(error)
    assert "cannot be converted" not in str(error)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_expression_failure_keeps_cause_without_cell_values(pyplot, tmp_path, monkeypatch, axis):
    # Undefined symbol: the cause (cexprtk ParseException) stays and holds no data.
    error, formatted = _failure(pyplot, tmp_path, monkeypatch, axis, pd.DataFrame({"v": [123456.789]}), expr="q+1")
    assert error.__cause__ is not None
    assert "123456.789" not in formatted


@pytest.mark.parametrize("values", [
    ["a", 1],
    [pd.Timestamp("2024-01-01")],
    [{"k": 1}],
])
def test_mixed_and_non_number_object_columns_fail(pyplot, tmp_path, monkeypatch, values):
    error, _ = _failure(pyplot, tmp_path, monkeypatch, "x", pd.DataFrame({"v": pd.Series(values, dtype=object)}))
    assert "cannot be converted to numbers" in str(error)


def test_object_column_of_decimals_with_null(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": pd.Series([decimal.Decimal("1.5"), None, decimal.Decimal("3")], dtype=object)})
    p = _run(pyplot, tmp_path, df, df, x_expr="x*2", monkeypatch=monkeypatch)
    assert p["x"].iloc[0] == 3.0 and pd.isna(p["x"].iloc[1]) and p["x"].iloc[2] == 6.0


def test_bool_column(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": [True, False, True]})
    p = _run(pyplot, tmp_path, df, df, x_expr="x+0.5", monkeypatch=monkeypatch)
    assert p["x"].tolist() == [1.5, 0.5, 1.5]


def test_nullable_int_missing_value_stays_missing(pyplot, tmp_path, monkeypatch):
    df = pd.DataFrame({"v": pd.Series([2, None, 4], dtype="Int64")})
    p = _run(pyplot, tmp_path, df, df, y_expr="y/4", monkeypatch=monkeypatch)
    assert p["y"].iloc[0] == 0.5 and pd.isna(p["y"].iloc[1]) and p["y"].iloc[2] == 1.0


@pytest.mark.parametrize("arrow_type", ["int64", "double"])
def test_arrow_backed_numeric_column(pyplot, tmp_path, monkeypatch, arrow_type):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"v": pd.Series([1, None, 3], dtype=f"{arrow_type}[pyarrow]")})
    p = _run(pyplot, tmp_path, df, df, x_expr="x/2", monkeypatch=monkeypatch)
    assert p["x"].iloc[0] == 0.5 and pd.isna(p["x"].iloc[1]) and p["x"].iloc[2] == 1.5
