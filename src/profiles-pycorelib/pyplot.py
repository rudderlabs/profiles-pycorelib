from .errors import MaterialNotFoundError, InvalidTransformationError
import cexprtk
import matplotlib.pyplot as plt
from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.logger import Logger
from profiles_rudderstack.schema import MaterializationBuildSpecSchema
from typing import List
import pandas as pd
import decimal
import numbers
import os
import re
import matplotlib
from matplotlib.backend_bases import FigureCanvasBase
matplotlib.use('Agg')

# A file extension such as ".png" or "png": letters and digits only, so it cannot hold a path.
EXTENSION_PATTERN = "^\\.?[A-Za-z0-9]+$"


def file_suffix(extension: str) -> str:
    """The extension with exactly one leading dot: "png" and ".png" both give ".png"."""
    return "." + extension.lstrip(".")


def is_supported_extension(extension) -> bool:
    """True if the value is a plain file extension of an image format matplotlib can write."""
    if not isinstance(extension, str) or re.fullmatch(EXTENSION_PATTERN, extension) is None:
        return False
    return extension.lstrip(".").lower() in FigureCanvasBase.get_supported_filetypes()


def output_file_name(material_name: str, extension: str) -> str:
    """File name of the plot: the material name followed by the model's extension.

    pb gives a material name without an extension, and matplotlib chooses the image format from
    the extension of the file it is asked to write.
    """
    suffix = file_suffix(extension)
    if material_name.lower().endswith(suffix.lower()):
        return material_name
    return material_name + suffix


def holds_numbers(column: pd.Series) -> bool:
    """True if the column is numeric or bool, or an object column whose non-null values are all numbers."""
    if pd.api.types.is_numeric_dtype(column.dtype):
        return True
    if column.dtype != object:
        return False
    return all(isinstance(v, (numbers.Real, decimal.Decimal)) for v in column.dropna())


def cast_to_float(df: pd.DataFrame, column_name: str, axis_name: str) -> None:
    """Turn a column into float64 so that a transformation can write fractional results into it.

    The error names the column and its dtype but never a cell value: pb reports the traceback in
    customer-visible logs, and the original conversion error quotes the value that failed.
    """
    column = df[column_name]
    if holds_numbers(column):
        try:
            df[column_name] = column.astype("float64")
            return
        except (TypeError, ValueError):
            pass
    raise InvalidTransformationError(
        f"{axis_name}-axis column '{column_name}' ({column.dtype}) cannot be converted to numbers"
    ) from None


class PyPlotModel(BaseModelType):
    TypeName = "pyplot"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "size": {"type": "string", "pattern": "^\\d+x\\d+$"},
            "grid": {"type": "boolean"},
            "extension": {"type": "string", "pattern": EXTENSION_PATTERN},
            "x_axis": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "column": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "transformation": {"type": "string"}
                },
                "required": ["label", "column", "inputs"],
                "additionalProperties": False
            },
            "y_axis": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "column": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "transformation": {"type": "string"}
                },
                "required": ["label", "column", "inputs"],
                "additionalProperties": False
            },
            **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["title", "extension", "x_axis", "y_axis"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        super().__init__(build_spec, schema_version, pb_version)

    def get_material_recipe(self) -> PyNativeRecipe:
        return PyPlotRecipe(
            self.build_spec.get("title"),
            self.build_spec.get("size", "8x8"),
            self.build_spec.get("grid", False),
            self.build_spec.get("x_axis"),
            self.build_spec.get("y_axis"),
            self.build_spec.get("extension"),
        )

    def validate(self):
        if self.schema_version < 51:
            return False, "schema version should >= 51"
        if not is_supported_extension(self.build_spec.get("extension")):
            supported = ", ".join("." + name for name in sorted(FigureCanvasBase.get_supported_filetypes()))
            return False, f"extension should be one of: {supported}"
        return True, "Validated successfully"

    def get_db_object_name_suffix(self):
        return file_suffix(self.build_spec.get("extension"))


class PyPlotRecipe(PyNativeRecipe):
    def __init__(self, title, size, grid, x_axis, y_axis, extension=".png") -> None:
        self.logger = Logger("pyplot_recipe")
        self.title = title
        self.size = size
        self.grid = grid
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.extension = extension

    def describe(self, this: WhtMaterial):
        description = this.name() + \
            " will be created to show the graphical representation of " + self.title + " ."
        return description, ".txt"

    def register_dependencies(self, this: WhtMaterial):
        for input in self.x_axis.get("inputs"):
            this.de_ref(input)
        for input in self.y_axis.get("inputs"):
            this.de_ref(input)

    def execute(self, this: WhtMaterial):
        tablesList: List[pd.DataFrame] = []
        output_folder = (this.get_output_folder())

        models = []
        for input_model in self.x_axis.get("inputs"):
            models.append(input_model)
        for input_model in self.y_axis.get("inputs"):
            models.append(input_model)

        for in_model in models:
            input_material = this.de_ref(in_model)
            if input_material is None:
                raise MaterialNotFoundError(
                    f"this.de_ref: unable to get material for {in_model}")
            material_df = input_material.get_df()
            tablesList.append(material_df)

        x_axis_dfs = tablesList[:(len(models)//2)]
        y_axis_dfs = tablesList[(len(models)//2):]

        df_x = pd.concat(x_axis_dfs, ignore_index=True)
        df_y = pd.concat(y_axis_dfs, ignore_index=True)

        if self.x_axis.get("transformation", False):
            # Integer columns cannot hold fractional results (TypeError on pandas 3).
            cast_to_float(df_x, self.x_axis.get("column"), "X")
            try:
                cexprtk.check_expression(self.x_axis.get("transformation"))
                for i in range(len(df_x[self.x_axis.get("column")])):
                    df_x.at[i, self.x_axis.get("column")] = cexprtk.evaluate_expression(
                        self.x_axis.get("transformation"), {"x": df_x.at[i, self.x_axis.get("column")]})
            except Exception as e:
                # cexprtk errors hold the expression, symbol names and a type name, never a cell value,
                # so the cause stays in the traceback.
                raise InvalidTransformationError(
                    f"Transformation of X-axis values cannot be done as invalid transformation-expression"
                    f" ({type(e).__name__})") from e

        if self.y_axis.get("transformation", False):
            # Integer columns cannot hold fractional results (TypeError on pandas 3).
            cast_to_float(df_y, self.y_axis.get("column"), "Y")
            try:
                cexprtk.check_expression(self.y_axis.get("transformation"))
                for i in range(len(df_y[self.y_axis.get("column")])):
                    df_y.at[i, self.y_axis.get("column")] = cexprtk.evaluate_expression(
                        self.y_axis.get("transformation"), {"y": df_y.at[i, self.y_axis.get("column")]})
            except Exception as e:
                # cexprtk errors hold the expression, symbol names and a type name, never a cell value,
                # so the cause stays in the traceback.
                raise InvalidTransformationError(
                    f"Transformation of Y-axis values cannot be done as invalid transformation-expression"
                    f" ({type(e).__name__})") from e

        os.makedirs(output_folder, exist_ok=True)

        height, width = map(int, self.size.split('x'))
        file_name = output_file_name(this.name(), self.extension)
        plotGraph(df_x[self.x_axis.get("column")],
                  df_y[self.y_axis.get("column")],
                  height, width, self.x_axis.get("label"),
                  self.y_axis.get("label"),
                  output_folder,
                  self.title,
                  file_name, self.grid)

def plotGraph(df_x, df_y, h, w, label_x, label_y, output_folder, title, file_name, grid):
    figure = plt.figure(figsize=(h, w))
    try:
        plt.plot(df_x, df_y, marker='o')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)
        output_file_path = os.path.join(output_folder, file_name)
        if grid:
            plt.grid()
        plt.savefig(output_file_path)
    finally:
        # The pb Python process stays alive for the whole run, so every figure has to be released.
        plt.close(figure)

    # matplotlib adds an extension to a file name that has none, so check the path that was asked for.
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"Plot was not saved at the specified location: {output_file_path}")
