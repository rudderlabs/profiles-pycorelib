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
import matplotlib.pyplot as plt
import cexprtk

def plotGraph(df_x, df_y,h,w,label_x, label_y, output_folder, title, file_name, grid):
    plt.figure(figsize=(h, w))
    plt.plot(df_x, df_y, marker='o')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    output_file_path = os.path.join(output_folder, file_name)
    if grid:
        plt.grid()
    plt.savefig(output_file_path)



class PyPlotModel(BaseModelType):
    TypeName = "pyplot"
    BuildSpecSchema= {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "size": {"type": "string", "pattern": "^\\d+x\\d+$"},
            "grid": {"type": "boolean"},
            "x_axis": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "column": {"type": "string"},
                "input": {"type": "string"},
                "transformation": {"type": "string"}
            },
            "required": ["label", "column", "input"],
            "additionalProperties": False
            },
            "y_axis": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "column": {"type": "string"},
                "input": {"type": "string"},
                "transformation": {"type": "string"}
            },
            "required": ["label", "column", "input"],
            "additionalProperties": False
            },
            **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["title", "x_axis", "y_axis"],
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

class PyPlotRecipe(PyNativeRecipe):
    def __init__(self, title, size, grid, x_axis, y_axis) -> None:
        self.title=title
        self.size=size
        self.grid=grid
        self.x_axis=x_axis
        self.y_axis=y_axis

    def describe(self, this: WhtMaterial):
        description=this.name()+ "is created"
        return description, ".txt"

    def prepare(self, this: WhtMaterial):
        self.logger.info("Preparing")
        this.de_ref(self.x_axis.get("input"))
        this.de_ref(self.y_axis.get("input"))

    def execute(self, this: WhtMaterial):
        tablesList: List[pd.DataFrame] = []
        output_folder= (this.get_output_folder())

        models=[]
        models.append(self.x_axis.get("input"))
        models.append(self.y_axis.get("input"))

        tablesList: List[pd.DataFrame] = []
        for in_model in models:
            input_material = this.de_ref(in_model)
            if input_material is None:
                self.logger.error("Input Matrial is Nil")
            df_or_iterator = input_material.get_df()
            if isinstance(df_or_iterator, pd.DataFrame):
                tablesList.append(df_or_iterator)
            elif isinstance(df_or_iterator, Iterator):
                tablesList.extend(df_or_iterator)

        df_x=tablesList[0]
        df_y=tablesList[1]

        if ("transformation" in self.x_axis) and self.x_axis.get("transformation"):
            try:
                cexprtk.check_expression(self.x_axis.get("transformation"))
                for i in range(len(df_x[self.x_axis.get("column")])):
                    df_x.at[i, self.x_axis.get("column")] = cexprtk.evaluate_expression(self.x_axis.get("transformation"), {"x": df_x.at[i, self.x_axis.get("column")]})
            except:
                self.logger.error("Transformation of X-axis values cannot be done as invalid transformation-expression")

        if ("transformation" in self.y_axis) and self.y_axis.get("transformation"):
            try:
                cexprtk.check_expression(self.y_axis.get("transformation"))
                for i in range(len(df_y[self.y_axis.get("column")])):
                    df_y.at[i, self.y_axis.get("column")] = cexprtk.evaluate_expression(self.y_axis.get("transformation"), {"y": df_y.at[i, self.y_axis.get("column")]})
            except:
                self.logger.error("Transformation of Y-axis values cannot be done as invalid transformation-expression")

        os.makedirs(output_folder, exist_ok=True)

        height, width = map(int, self.size.split('x'))
        file_name=this.name()

        plotGraph(df_x[self.x_axis.get("column")], df_y[self.y_axis.get("column")], height,width ,self.x_axis.get("label"), self.y_axis.get("label"),output_folder, self.title, file_name,self.grid )