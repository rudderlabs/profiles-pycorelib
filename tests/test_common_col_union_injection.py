"""Regression tests for build-spec injection in common column union templates."""
from types import SimpleNamespace

import pytest


_BREAKOUT = "inputs/x') %}{{ warehouse.exec_sql('x') }}{% with q=this.DeRef('y"
_COLUMNS = [{"name": "user_id", "type": "varchar"}, {"name": "timestamp", "type": "timestamp"}]


class _FakeMaterial:
    def __init__(self, columns, captured):
        self._columns = columns
        self._captured = captured
        self.wht_ctx = SimpleNamespace(
            is_null_ctx=False,
            client=SimpleNamespace(
                query_sql_without_result=lambda sql: captured.setdefault("sql", sql)
            ),
        )

    def de_ref(self, in_model, dependency=None):
        return SimpleNamespace(get_columns=lambda: self._columns)

    def execute_text_template(self, template):
        self._captured["template"] = template
        return "compiled sql"

    def name(self):
        return "test_material"


def test_input_breakout_rejected_before_template_and_sql(common_col_union):
    for method in ("execute", "register_dependencies"):
        captured = {}
        recipe = common_col_union.CommonColumnUnionRecipe([_BREAKOUT])

        with pytest.raises(ValueError):
            getattr(recipe, method)(_FakeMaterial(_COLUMNS, captured))

        assert captured == {}


def test_valid_inputs_complete(common_col_union):
    captured = {}
    recipe = common_col_union.CommonColumnUnionRecipe(["inputs/tbl_a", "inputs/tbl_b"])

    recipe.execute(_FakeMaterial(_COLUMNS, captured))

    assert "this.DeRef('inputs/tbl_a')" in captured["template"]
    assert "user_id" in captured["template"]
    assert captured["sql"] == "compiled sql"


def test_column_breakout_rejected(common_col_union):
    captured = {}
    columns = [{"name": "user_id') %}{{ x }}{% exec %}", "type": "varchar"}]
    recipe = common_col_union.CommonColumnUnionRecipe(["inputs/tbl_a"])

    with pytest.raises(ValueError):
        recipe.execute(_FakeMaterial(columns, captured))

    assert captured == {}


def test_validate_model_ref(common_col_union):
    validate = common_col_union.validate_model_ref
    for ref in (
        "inputs/tbl_a",
        "models/input_table_size",
        "entity/user/var_table",
        "user/all/var_table",
    ):
        assert validate(ref) == ref

    for ref in (
        _BREAKOUT,
        "inputs/x'",
        "a{% exec %}",
        "a{{b}}",
        "a b",
        "",
        "/leading",
        "trailing/",
        "a//b",
    ):
        with pytest.raises(ValueError):
            validate(ref)
