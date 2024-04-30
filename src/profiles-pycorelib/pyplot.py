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
from typing import Iterator
import os
import matplotlib
matplotlib.use('Agg')

class PyPlotModel(BaseModelType):
    TypeName = "pyplot"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "size": {"type": "string", "pattern": "^\\d+x\\d+$"},
            "grid": {"type": "boolean"},
            "extension": {"type": "string"},
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
        )

    def validate(self):
        return self.schema_version >= 51, "schema version should >= 51"

    def get_db_object_name_suffix(self):
        return self.build_spec.get("extension")


class PyPlotRecipe(PyNativeRecipe):
    def __init__(self, title, size, grid, x_axis, y_axis) -> None:
        self.logger = Logger("pyplot_recipe")
        self.title = title
        self.size = size
        self.grid = grid
        self.x_axis = x_axis
        self.y_axis = y_axis

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
            try:
                cexprtk.check_expression(self.x_axis.get("transformation"))
                for i in range(len(df_x[self.x_axis.get("column")])):
                    df_x.at[i, self.x_axis.get("column")] = cexprtk.evaluate_expression(
                        self.x_axis.get("transformation"), {"x": df_x.at[i, self.x_axis.get("column")]})
            except:
                raise InvalidTransformationError(
                    "Transformation of X-axis values cannot be done as invalid transformation-expression")

        if self.y_axis.get("transformation", False):
            try:
                cexprtk.check_expression(self.y_axis.get("transformation"))
                for i in range(len(df_y[self.y_axis.get("column")])):
                    df_y.at[i, self.y_axis.get("column")] = cexprtk.evaluate_expression(
                        self.y_axis.get("transformation"), {"y": df_y.at[i, self.y_axis.get("column")]})
            except:
                raise InvalidTransformationError(
                    "Transformation of Y-axis values cannot be done as invalid transformation-expression")

        os.makedirs(output_folder, exist_ok=True)

        height, width = map(int, self.size.split('x'))
        file_name = this.name()
        plotGraph(df_x[self.x_axis.get("column")],
                  df_y[self.y_axis.get("column")],
                  height, width, self.x_axis.get("label"),
                  self.y_axis.get("label"),
                  output_folder,
                  self.title,
                  file_name, self.grid)

def plotGraph(df_x, df_y, h, w, label_x, label_y, output_folder, title, file_name, grid):
    plt.figure(figsize=(h, w))
    plt.plot(df_x, df_y, marker='o')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    output_file_path = os.path.join(output_folder, file_name)
    if grid:
        plt.grid()
    plt.savefig(output_file_path)

    if not os.path.exists(output_file_path):
        raise FileNotFoundError(
            "Error: Plot was not saved at the specified location:", output_file_path)
