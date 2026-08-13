"""Shared test fixtures for profiles-pycorelib.

The model modules import ``profiles_rudderstack`` at module scope. That package
talks to the ``pb`` runtime over a local gRPC tunnel and is not installable in a
plain unit-test environment, so we stub the handful of symbols the models import.
Everything else (pandas, numpy, scipy, matplotlib, seaborn, plotly) is the real
library, so the code under test runs unmodified.
"""
import importlib.util
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "src" / "profiles-pycorelib"


def _install_profiles_rudderstack_stubs() -> None:
    if "profiles_rudderstack" in sys.modules:
        return

    root = types.ModuleType("profiles_rudderstack")
    sys.modules["profiles_rudderstack"] = root

    model = types.ModuleType("profiles_rudderstack.model")

    class BaseModelType:
        def __init__(self, build_spec, schema_version, pb_version):
            self.build_spec = build_spec
            self.schema_version = schema_version
            self.pb_version = pb_version

    model.BaseModelType = BaseModelType
    sys.modules["profiles_rudderstack.model"] = model

    schema = types.ModuleType("profiles_rudderstack.schema")
    for name in (
        "ContractBuildSpecSchema",
        "EntityKeyBuildSpecSchema",
        "EntityIdsBuildSpecSchema",
        "MaterializationBuildSpecSchema",
    ):
        setattr(schema, name, {"properties": {}})
    sys.modules["profiles_rudderstack.schema"] = schema

    recipe = types.ModuleType("profiles_rudderstack.recipe")

    class PyNativeRecipe:
        pass

    recipe.PyNativeRecipe = PyNativeRecipe
    sys.modules["profiles_rudderstack.recipe"] = recipe

    material = types.ModuleType("profiles_rudderstack.material")

    class WhtMaterial:
        pass

    material.WhtMaterial = WhtMaterial
    sys.modules["profiles_rudderstack.material"] = material

    logger = types.ModuleType("profiles_rudderstack.logger")

    class Logger:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warn(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logger.Logger = Logger
    sys.modules["profiles_rudderstack.logger"] = logger


def _load_model_module(module_name: str):
    _install_profiles_rudderstack_stubs()
    path = _MODELS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def attribution_model():
    """The real ``attribution_model`` module, loaded from source."""
    return _load_model_module("attribution_model")
