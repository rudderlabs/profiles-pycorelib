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
import threading

def plotGraph(df_x, df_y,h,w,label_x, label_y, output_folder, title, img_name):
    plt.figure(figsize=(h, w))
    plt.plot(df_x, df_y, marker='o')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    output_file_path1 = os.path.join(output_folder, 'total_ids_vs_credits_spent.png')
    plt.savefig(output_file_path1)



class PyPlotModel(BaseModelType):
    TypeName = "pyplot"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "num_of_graphs": {"type": "integer", "minimum": 1},
            "fig_height": {"type": "array", "items": {"type": "integer"}, "minItems":1},
            "fig_width": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
            "label_x": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "label_y": {"type": "array", "items": {"type": "string"}, "minItems":1},
            "title": {"type": "array", "items": {"type": "string"}, "minItems":1},
            "output_folder": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "col_x": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "col_y": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "tables": {"type": "array","items": {"type": "array","items": {"type":"string"},"minItems": 2,"maxItems": 2},"minItems":  1},
            "img_name": {"type": "array", "items": {"type": "string"}, "minItems": 1},

             **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["num_of_graphs", "fig_height", "fig_width", "label_x", "label_y", "title", "output_folder", "col_x", "col_y", "tables"],
        "additionalProperties": False
        }
    
    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        super().__init__(build_spec, schema_version, pb_version)
        print(build_spec)

    def get_material_recipe(self) -> PyNativeRecipe:
        return PyPlotRecipe(self.build_spec["num_of_graphs"],self.build_spec["fig_height"],
            self.build_spec["fig_width"],
              self.build_spec["label_x"],
            self.build_spec["label_y"],
            self.build_spec["title"],
            self.build_spec["output_folder"],
            self.build_spec["col_x"],
            self.build_spec["col_y"],
            self.build_spec["tables"],
            self.build_spec["img_name"]
            )

    def validate(self):
        return self.schema_version >= 51, "schema version should >= 51"

class PyPlotRecipe(PyNativeRecipe):
    def __init__(self, num_of_graphs: int, fig_height: List[int], fig_width: List[int], label_x: List[str], label_y: List[str], title: List[str], output_folder: List[str],col_x: List[str], col_y: List[str], tables: List[List[str]], img_name: List[str]) -> None:
        self.logger = Logger("graph_recipe")
        self.sql = ""

        self.num_of_graphs=num_of_graphs
        self.fig_height=fig_height
        self.fig_width=fig_width
        self.label_x=label_x
        self.label_y=label_y
        self.title=title
        self.output_folder=output_folder
        self.col_x=col_x
        self.col_y=col_y
        self.tables = tables
        self.img_name=img_name

        print(self.num_of_graphs)
        print(self.fig_height)
        print(self.fig_width)
        print(self.label_x)
        print(self.label_y)
        print(self.title)
        print(self.output_folder)
        print(self.col_x)
        print(self.col_y)
        print(self.tables)
        print(self.img_name)
        

    def describe(self, this: WhtMaterial):
        return self.sql, ".txt"

    def prepare(self, this: WhtMaterial):
        self.logger.info("Preparing")
        for table in self.tables:
            for in_model in table:
                this.de_ref(in_model)
 

    def execute(self, this: WhtMaterial):
        self.logger.info("Executing PyPlotModel")
        count=0
        for table in self.tables:
            output_path=""
            tablesList: List[pd.DataFrame] = []
            for in_model in table:
                input_material = this.de_ref(in_model)
                if input_material is None:
                    self.logger.info(
                        "Input material for {0} not found".format(in_model))
                    continue
                else:
                    output_path=input_material.get_output_folder()
                    self.logger.info(
                        "Inside graph - Name of the material is: {0}".format(input_material.name()))

                df_or_iterator = input_material.get_df()

                if isinstance(df_or_iterator, pd.DataFrame):
                    tablesList.append(df_or_iterator)
                elif isinstance(df_or_iterator, Iterator):
                    tablesList.extend(df_or_iterator)
                
            df1=tablesList[0]
            df2=tablesList[1]
            output_folder=output_path+"/"+self.output_folder[count]
            os.makedirs(output_folder, exist_ok=True)



            df1[self.col_x[count]] = (df1[self.col_x[count]].astype(float) /1000).round().astype(int)
            df2[self.col_y[count]]=(df2[self.col_y[count]].astype(float) / 3600).round(4)

            count=count+1

            plotGraph(df1[self.col_x[count]], df2[self.col_y[count]], self.fig_height[count],self.fig_width[count],self.label_x[count], self.label_y[count],output_folder, self.title[count], self.img_name[count] )