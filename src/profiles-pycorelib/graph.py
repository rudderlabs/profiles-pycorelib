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


class GraphModel(BaseModelType):
    TypeName = "graph"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "inputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["inputs"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        super().__init__(build_spec, schema_version, pb_version)

    def get_material_recipe(self) -> PyNativeRecipe:
        return GraphRecipe(self.build_spec["inputs"])

    def validate(self):
        return self.schema_version >= 51, "schema version should >= 51"


class GraphRecipe(PyNativeRecipe):
    def __init__(self, inputs: List[str]) -> None:
        self.inputs = inputs
        self.logger = Logger("graph_recipe")
        self.sql = ""

    def describe(self, this: WhtMaterial):
        return self.sql, ".txt"

    def prepare(self, this: WhtMaterial):
        self.logger.info("Preparing")
        for in_model in self.inputs:
            this.de_ref(in_model)
 

    def execute(self, this: WhtMaterial):
        self.logger.info("Executing GraphModel ")
        tables: List[pd.DataFrame] = []
        output_folder=""
        for in_model in self.inputs:
            input_material = this.de_ref(in_model)
            if input_material is None:
                self.logger.info(
                    "Input material for {0} not found".format(in_model))
                continue
            else:
                output_folder=input_material.get_output_folder()
                self.logger.info(
                    "Inside graph - Name of the material is: {0}".format(input_material.name()))

            df_or_iterator = input_material.get_df()

            if isinstance(df_or_iterator, pd.DataFrame):
                tables.append(df_or_iterator)
            elif isinstance(df_or_iterator, Iterator):
                tables.extend(df_or_iterator)
                
        df1=tables[0]
        df2=tables[1]
        df3=tables[2]

        output_folder=output_folder+"/reports"
        print(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        df1['total_records'] = (df1['total_records'].astype(float) / 1000).round().astype(int)
        df1['post_stitched_ids'] = (df1['post_stitched_ids'].astype(float) / 1000).round().astype(int)
        df2['run_time_in_sec']=(df2['run_time_in_sec'].astype(float) / 3600).round(4)
        df3['run_time_in_sec']=(df3['run_time_in_sec'].astype(float) / 3600).round(4)

        def plot_graph(output_folder, df1, df2, df3):
            plt.figure(figsize=(8, 6))
            plt.plot(df1['total_records'], df2['run_time_in_sec'], marker='o')
            plt.xlabel('Total IDs (in thousands)')
            plt.ylabel('Credits Spent')
            plt.title('Total IDs vs. Credits Spent')
            plt.locator_params(axis='x', integer=True)
            output_file_path1 = os.path.join(output_folder, 'total_ids_vs_credits_spent.png')
            plt.savefig(output_file_path1)

            plt.figure(figsize=(8, 6))  
            plt.plot(df1['post_stitched_ids'], df3['run_time_in_sec'], marker='o')
            plt.xlabel('Total Users (in thousands)')
            plt.ylabel('Credits Spent')
            plt.title('Total Users vs. Credits Spent')
            plt.locator_params(axis='x', integer=True)
            output_file_path2 = os.path.join(output_folder, 'total_users_vs_credits_spent.png')
            plt.savefig(output_file_path2)

        threading.Thread(target=plot_graph, args=(output_folder, df1, df2, df3)).start()