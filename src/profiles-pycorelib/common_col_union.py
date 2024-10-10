from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.logger import Logger
from typing import List

class CommonColumnUnionModel(BaseModelType):
    TypeName = "common_column_union"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "inputs": { "type": "array", "items": { "type": "string" } },            
        },
        "required": ["inputs"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        super().__init__(build_spec, schema_version, pb_version)

    def get_material_recipe(self)-> PyNativeRecipe:
        return CommonColumnUnionRecipe(self.build_spec["inputs"])

    def validate(self):
        # Model Validate
        if self.build_spec.get("inputs") is None or len(self.build_spec["inputs"]) == 0:
            return False, "inputs are required"
        
        return super().validate()


class CommonColumnUnionRecipe(PyNativeRecipe):
    def __init__(self, inputs: List[str]) -> None:
        self.inputs = inputs
        self.logger = Logger("common_column_union_recipe")
        self.sql = ""

    def describe(self, this: WhtMaterial):
        return self.sql, ".sql"

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
                    select  <determined at runtime> from {{{{input_mat}}}}
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
            columns = in_material.get_columns()
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
            self.logger.error("error executing common_column_union_recipe, sql is empty")
        
        this.wht_ctx.client.query_sql_without_result(self.sql)