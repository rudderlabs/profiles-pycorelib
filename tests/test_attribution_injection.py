"""Regression tests for pandas ``df.query()`` code injection in the
``campaign_attribution_scores`` model (RUD-3028 / HackerOne #3912408).

A build-spec field was interpolated, unescaped, into ``DataFrame.query()``.
pandas resolves ``@name`` against module globals, and the module imports ``os``,
so ``@os.system(...)`` in a config field executed arbitrary shell commands in the
pb runner process. These tests prove such a payload is rejected, not executed.
"""
import pathlib

import pandas as pd
import pytest


class _FakeRef:
    def __init__(self, df):
        self._df = df

    def get_df(self):
        return self._df


class _FakeMaterial:
    """Minimal stand-in for WhtMaterial covering what execute() touches."""

    def __init__(self, df, output_folder):
        self._df = df
        self._output_folder = str(output_folder)

    def de_ref(self, _key):
        return _FakeRef(self._df.copy())

    def get_output_folder(self):
        return self._output_folder

    def name(self):
        return "test_material"

    def write_output(self, _df):
        return None


def _input_df():
    return pd.DataFrame(
        {
            "days_since_first_seen": [1, 5, 40],
            "campaign": ["email", "ads", "social"],
            "converted": [1, 0, 1],
        }
    )


def test_days_since_first_seen_var_injection_is_rejected(attribution_model, tmp_path):
    # execute() lower-cases the field, so keep the sentinel path lower-case too.
    sentinel = pathlib.Path(str(tmp_path / "pwned.txt").lower())
    payload = f'days_since_first_seen+@os.system("touch {sentinel}")'
    config = {
        "entity": "user",
        "touchpoint_var": "campaign",
        "conversion_entity_var": "converted",
        "days_since_first_seen_var": payload,
        "first_seen_since": 100,
        "enable_visualisation": False,
    }
    recipe = attribution_model.AttributionModelRecipe(config)
    material = _FakeMaterial(_input_df(), tmp_path)

    # The fixed code must reject the non-identifier column name with a ValueError.
    with pytest.raises(ValueError):
        recipe.execute(material)

    assert not sentinel.exists(), (
        "injected shell command executed via df.query() — RCE not fixed"
    )


def test_markov_conversion_col_injection_is_rejected(attribution_model, tmp_path):
    sentinel = tmp_path / "pwned_markov.txt"
    payload = f'converted+@os.system("touch {sentinel}")'
    df = pd.DataFrame(
        {
            "touchpoints": [["email", "ads"], ["social"]],
            "converted": [1, 0],
        }
    )
    models = attribution_model.MultiTouchModels(logger=attribution_model.Logger("t"))

    with pytest.raises(ValueError):
        models.get_markov_attribution(df, payload, "touchpoints", str(tmp_path), False)

    assert not sentinel.exists(), "injected shell command executed via markov df.query()"


def test_validate_column_name_accepts_identifiers_and_rejects_expressions(attribution_model):
    df = _input_df()
    validate = attribution_model._validate_column_name

    assert validate("days_since_first_seen", df) == "days_since_first_seen"

    for bad in [
        'days_since_first_seen+@os.system("id")',
        "@os",
        "a; b",
        "col name",
        "days_since_first_seen ",
        "1col",
        "nonexistent_column",
    ]:
        with pytest.raises(ValueError):
            validate(bad, df)


def test_valid_config_filters_by_days_and_completes(attribution_model, tmp_path):
    captured = {}

    class _CapturingMaterial(_FakeMaterial):
        def write_output(self, df):
            captured["output"] = df

    df = pd.DataFrame(
        {
            "days_since_first_seen": [1, 5, 8, 40, 60],
            "campaign": ["email", "ads", "email,ads", "social", "social,email"],
            "converted": [1, 0, 1, 1, 0],
        }
    )
    config = {
        "entity": "user",
        "touchpoint_var": "campaign",
        "conversion_entity_var": "converted",
        "days_since_first_seen_var": "days_since_first_seen",
        "first_seen_since": 10,
        "enable_visualisation": False,
    }
    recipe = attribution_model.AttributionModelRecipe(config)

    recipe.execute(_CapturingMaterial(df, tmp_path))

    touchpoints = set(captured["output"]["campaign"])
    assert "email" in touchpoints
    # rows with days_since_first_seen > 10 (the "social" touchpoints) are filtered out
    assert "social" not in touchpoints
