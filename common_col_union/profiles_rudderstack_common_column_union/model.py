from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.contract import BuildContract
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.utils import GetLogger
from typing import List


# Model Type Definition
class CommonColumnUnionModel(BaseModelType):
    TypeName = "common_column_union"

    # Specify the expected model_spec structure using jsonschema
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "inputs": { "type": "array", "items": { "type": "string" } },            
        },
        "required": ["inputs"],
        "additionalProperties": False
    }

    def __init__(self, buildSpec: dict, schemaVersion: int, pbVersion: str) -> None:
        super().__init__(buildSpec, schemaVersion, pbVersion)

    def GetMaterialRecipe(self)-> PyNativeRecipe:
        return CommonColumnUnionRecipe(self.buildSpec["inputs"])

    def Validate(self):
        # Model Validate
        if self.buildSpec.get("inputs") is None or len(self.buildSpec["inputs"]) == 0:
            return False, "inputs are required"
        
        return super().Validate()


# Recipe Definition 
class CommonColumnUnionRecipe(PyNativeRecipe):
    def __init__(self, inputs: List[str]) -> None:
        self.inputs = inputs
        self.logger = GetLogger(__name__)

    # This is to add debug information that will be written to output folder in compile, in other models the output is sql
    # In py native model we have the control on that the output should be
    def Describe(self, this: WhtMaterial):
        materialName = this.Name()
        return f"""Material - {materialName}\nInputs: {self.inputs}""", ".txt" # content, extension


    # Prepare ensure that all model dependencies are available, by calling this.DeRef we add them to material's dependency list
    def Prepare(self, this: WhtMaterial):
        for inModel in self.inputs:
            # Can specify input contract here(currently, optional)
            # contract = BuildContract('{ "is_event_stream": true, "with_columns":[{"name":"num"}] }')
            # this.DeRef(inModel, contract)
            this.DeRef(inModel)

    def Execute(self, this: WhtMaterial):
        tables = []
        common_columns_count = {}
        for inModel in self.inputs:
            inputMaterial = this.DeRef(inModel)
            tables.append(inputMaterial.Name())

            columns = inputMaterial.GetColumns()
            for col in columns:
                key = (col["name"], col["type"])
                if key in common_columns_count:
                    common_columns_count[key] += 1
                else:
                    common_columns_count[key] = 1
        
        common_columns = [name for (name, _), count in common_columns_count.items() if count == len(self.inputs)]

        if len(common_columns) == 0:
            self.logger.error("No common columns found")
            return
        
        select_columns = ', '.join([f'{column}' for column in common_columns])
        union_queries = []
        for table in tables:
            union_queries.append(f"SELECT {select_columns} FROM {table}")
        
        union_sql = " UNION ALL ".join(union_queries)

        # Execute the template for union query
        this.whtCtx.client.QueryTemplateWithoutResult(
            "{% macro selector_sql() %}" + 
            union_sql + 
            "{% endmacro %}" + 
            """{% exec %}{{ warehouse.CreateReplaceTableAs(this.Name(), selector_sql()) }}{% endexec %}"""
        )
        
