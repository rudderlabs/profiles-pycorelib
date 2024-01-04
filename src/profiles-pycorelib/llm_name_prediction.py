from profiles_rudderstack.model import BaseModelType
from profiles_rudderstack.contract import build_contract
from profiles_rudderstack.recipe import PyNativeRecipe
from profiles_rudderstack.material import WhtMaterial
from profiles_rudderstack.logger import Logger
from typing import List
import os
import time
import pandas as pd
import json 
import hashlib
from langchain.chains import ConversationChain
from langchain.llms import OpenAI, Bedrock, LlamaCpp
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_google_genai import GoogleGenerativeAI

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass


class LLMNamePredictionModel(BaseModelType):
    TypeName = "llm_name_prediction"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "entity_key": {"type": "string"},
            "inputs": {"type": "array", "items": { "type": "string"}},
            "prompt": {"type": "string"},
            "endpoint": {"type": "string"},
            "model": {"type": "string"}
        },
        "required": ["inputs","prompt","endpoint","model"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        super().__init__(build_spec, schema_version, pb_version)

    #  Specify the output contract for the model (optional)
    # def get_contract(self) -> Contract:
    #     return build_contract('{ "is_event_stream": false, "with_columns":[{"name":"num"}] }')
    
    #  Specify the entity key for the model (optional)
    # def get_entity_key(self):
    #     return self.build_spec.get("entity_key") # getting it from model spec
    #     or
    #     return "entity_key"
    
    def get_material_recipe(self)-> PyNativeRecipe:
        return LLMNamePredictionRecipe(self.build_spec.get("inputs"),self.build_spec.get("prompt"),self.build_spec.get("endpoint"),self.build_spec.get("model"))

    def validate(self):
        # Model Validate
        if self.build_spec.get("inputs") is None or len(self.build_spec["inputs"]) == 0:
            return False, "Property input is required"
        if self.build_spec.get("prompt") is None or len(self.build_spec["prompt"]) == 0:
            return False, "Property prompt is required"
        if self.build_spec.get("endpoint") is None or len(self.build_spec["endpoint"]) == 0:
            return False, "Property endpoint is required"
        if self.build_spec.get("model") is None or len(self.build_spec["model"]) == 0:
            return False, "Property model is required"

        return super().validate()


class LLMNamePredictionRecipe(PyNativeRecipe):
    def __init__(self, inputs: List[str], prompt: str, endpoint: str, model: str) -> None:
        self.inputs = inputs
        self.prompt = prompt
        self.endpoint = endpoint
        self.model = model
        self.logger = Logger("LLMNamePredictionRecipe")

    def describe(self, this: WhtMaterial):
        material_name = this.name()
        return f"""Material - {material_name}\nInputs: {self.inputs}\nPrompt: {self.prompt}\nEndpoint: {self.endpoint}\nModel: {self.model}""", ".txt"

    def prepare(self, this: WhtMaterial):
        for in_model in self.inputs:
            contract = build_contract('{ "is_event_stream": true, "with_columns":[{"name":"user_main_id"},{"name":"first_name"}] }')
            this.de_ref(in_model, contract)
        
        
    def execute(self, this: WhtMaterial):
        self.logger.info("Executing LLMNamePredictionRecipe:New")

        id_response_list = []
        hash_response_list = []

        tables : List[pd.DataFrame] = []
        in_model = self.inputs[0]
        input_material = this.de_ref(in_model)
        # df = input_material.get_table_data()
        # tables.append(df)
        
        # read data in batches, only supported in case of snowflake currently
        dfIter = input_material.get_table_data_batches()
        for batch in dfIter:
            batch = batch.dropna() #drop rows with null values, which can only be for null first name list
            #self.logger.info("Batch: {0}".format(batch))
            for index, row in batch.iterrows():
                first_name_list = row[2].replace("[","").replace("]","").replace("\n","").replace(" ","")
                if len(first_name_list)>0 and ("," in first_name_list): #consider only non-blank values

                    llm = Bedrock(region_name="us-east-1", model_id="anthropic.claude-v2") # default LLM

                    # init LLM based on endpoint and model

                    match self.endpoint.lower():
                        case "bedrock":
                            llm = Bedrock(region_name="us-east-1", model_id=self.model.lower()) # LLM init

                        case "openai":
                            llm = ChatOpenAI(temperature=0.2,model_name=self.model.lower())

                        case "google":
                            llm = GoogleGenerativeAI(model=self.model.lower())

                    chain = ConversationChain(llm = llm) # chain init

                    # query construction


                    # Prompt construction

                    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

                    Current conversation:
                    {history}
                    Human: {input}
                    AI Assistant:"""

                    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

                    # conversation construction
                    conversation = ConversationChain(
                        prompt=PROMPT,
                        llm=llm,
                        verbose=False,
                        memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
                    )

                    #delay to avoid rate limit
                    time.sleep(5)

                    complete_prompt = self.prompt + ": " + first_name_list

                    result = ""

                    prompt_hash = hashlib.sha1(complete_prompt.encode("utf-8")).hexdigest()

                    # check cache before invoking llm
                    try:
                        cache_retrieval_query = "select response from llm_response_cache where hash = '" + str(prompt_hash) + "'"
                        cached_response_df = this.wht_ctx.client.query_sql_with_result(cache_retrieval_query)
                        result = str(cached_response_df.iloc[:,0][0])
                        self.logger.info("Result from cache: " + result)
                    except Exception as e: # unable to retrieve data (table/row does not exist or other)
                        result = conversation.predict(input = complete_prompt)
                        result = result.replace(".","").split()[-1]
                        self.logger.info("Result from LLM: " + result)

                    #self.logger.info(row[0] + " : " + first_name_list + " : " + result)

                    id_response = {}
                    id_response["user_main_id"] = row[0]
                    id_response["cleaned_first_name"] = result
                    id_response_list.append(id_response)

                    hash_response = {}
                    hash_response["hash"] = prompt_hash
                    hash_response["response"] = result
                    hash_response_list.append(hash_response) 

        # Get model's entity, entity will be none if no entity_key is provided
        """
        entity = input_material.entity()
        if entity is not None:
            name = entity.get("Name")
            id_types = entity.get("IdTypes")
            main_id_type = entity.get("MainIdType")
            id_column_name = entity.get("IdColumnName")
            self.logger.info(json.dumps(entity))

        """
        # If you want to get only the columns that are required, you can use the following code
        # df = input_material.get_table_data(select_columns=["user_main_id","first_name"])

        # you can also query with result using
        # df = this.wht_ctx.client.query_sql_with_result("select user_main_id as user_main_id, first_name as first_name from {0}".format(input_material.name()))
        # self.logger.info(str(df.columns))
        # Get output folder path
        # folder = this.get_output_folder()
        id_response_df = pd.DataFrame(id_response_list)
        hash_response_df = pd.DataFrame(hash_response_list)

        this.wht_ctx.client.write_df_to_table(id_response_df, "user_clean_name")

        this.wht_ctx.client.write_df_to_table(hash_response_df, "llm_response_cache") 

        self.logger.info("Finished executing LLMNamePredictionRecipe:New")

        
