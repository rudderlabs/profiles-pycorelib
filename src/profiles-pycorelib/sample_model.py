from typing import List

from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.schema import ContractBuildSpecSchema, EntityKeyBuildSpecSchema, EntityIdsBuildSpecSchema, MaterializationBuildSpecSchema
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.logger import Logger


class SampleCommonColUnionModel(BaseModelType):
    TypeName = "sample_common_col_union" # the name of the model type

    # json schema for the build spec
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "inputs": { "type": "array", "items": { "type": "string" } },
            # extend your schema from pre-defined schemas
            **ContractBuildSpecSchema["properties"],
            **EntityKeyBuildSpecSchema["properties"],
            **EntityIdsBuildSpecSchema["properties"],
            **MaterializationBuildSpecSchema["properties"],
        },
        "required": ["inputs"] + EntityKeyBuildSpecSchema["required"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        # call the base class constructor
        # it will initialize contract, entity_key, ids and materialization, if present in the build_spec
        super().__init__(build_spec, schema_version, pb_version)

        # override the values from the build_spec
        # self.entity_key = "user"

    def get_material_recipe(self)-> PyNativeRecipe:
        # return the recipe object
        return SampleCommonColumnUnionRecipe(self.build_spec["inputs"])


class SampleCommonColumnUnionRecipe(PyNativeRecipe):
    def __init__(self, inputs: List[str]) -> None:
        self.inputs = inputs
        self.logger = Logger("sample_recipe") # create a logger for debug/error logging
        self.sql = "" # the sql code to be executed

    # describe the compile output of the recipe - the sql code and the extension of the file
    def describe(self, this: WhtMaterial):
        return self.sql, ".sql"

    # prepare the material for execution - de_ref the inputs and create the sql code
    def register_dependencies(self, this: WhtMaterial):
        is_null_ctx = this.wht_ctx.is_null_ctx
        if is_null_ctx:
            for in_model in self.inputs:
                this.de_ref(in_model, dependency="optional")
            return

        inputs = [] # enabled inputs
        common_columns_count = {}
        for in_model in self.inputs:
            in_material = this.de_ref(in_model, dependency="optional")
            if in_material is None:
                continue # disabled

            inputs.append(in_model)
           
        union_sql = ""
        union_queries = []
        for in_model in inputs:
            union_queries.append(
            f"""{{% with input_mat = this.DeRef('{in_model}') %}}
                    select <determined at runtime> from {{{{input_mat}}}}
                {{% endwith %}}"""
            )

        union_sql = " UNION ALL ".join(union_queries)

        sql = this.execute_text_template(
            f"""
            {{% macro begin_block() %}}
                {{% macro selector_sql() %}}
                    {union_sql}
                {{% endmacro %}}
                {{% exec %}} {{{{warehouse.CreateReplaceTableAs(this, selector_sql())}}}} {{% endexec %}}
            {{% endmacro %}}

            {{% exec %}} {{{{warehouse.BeginEndBlock(begin_block())}}}} {{% endexec %}}"""
        )
        self.sql = sql

    # execute the material
    def execute(self, this: WhtMaterial):
        is_null_ctx = this.wht_ctx.is_null_ctx
        if is_null_ctx:
            for in_model in self.inputs:
                this.de_ref(in_model, dependency="optional")
            return

        inputs = [] # enabled inputs
        common_columns_count = {}
        for in_model in self.inputs:
            in_material = this.de_ref(in_model, dependency="optional")
            if in_material is None:
                continue # disabled

            inputs.append(in_model)
            columns = in_material.get_columns() # get the columns of the material
            for col in columns:
                key = (col["name"], col["type"])
                if key in common_columns_count:
                    common_columns_count[key] += 1
                else:
                    common_columns_count[key] = 1

        common_columns = [name for (name, _), count in common_columns_count.items() if count == len(inputs)]
        union_sql = ""
        if len(common_columns) > 0:
            select_columns = ', '.join([f"""{column}{{% if warehouse.DatabaseType() == "redshift" || warehouse.DatabaseType() == "snowflake" %}}::{{{{warehouse.DataType("timestamp")}}}}{{% endif %}}""" if column == "timestamp" else f'{column}' for column in common_columns])
            union_queries = []
            for in_model in inputs:
               union_queries.append(
                f"""{{% with input_mat = this.DeRef('{in_model}') %}}
                        select {select_columns} from {{{{input_mat}}}}
                    {{% endwith %}}"""
                )

            union_sql = " UNION ALL ".join(union_queries)
        else:
            union_queries = []
            for in_model in inputs:
               union_queries.append(
                f"""{{% with input_mat = this.DeRef('{in_model}') %}}
                        select * from {{{{input_mat}}}}
                    {{% endwith %}}"""
                )

            union_sql = " UNION ALL ".join(union_queries)

        sql = this.execute_text_template(
            f"""
            {{% macro begin_block() %}}
                {{% macro selector_sql() %}}
                    {union_sql}
                {{% endmacro %}}
                {{% exec %}} {{{{warehouse.CreateReplaceTableAs(this, selector_sql())}}}} {{% endexec %}}
            {{% endmacro %}}

            {{% exec %}} {{{{warehouse.BeginEndBlock(begin_block())}}}} {{% endexec %}}"""
        )
        self.sql = sql
        if self.sql == "":
            model_name = this.model.name()
            self.logger.error(f"error executing {model_name} model, compiled sql is empty")

        # execute the sql code
        this.wht_ctx.client.query_sql_without_result(self.sql)
