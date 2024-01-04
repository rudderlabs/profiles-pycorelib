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
from langchain.llms import Bedrock
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI


class LLMNamePredictionModel(BaseModelType):
    TypeName = "llm_name_prediction"
    BuildSpecSchema = {
        "type": "object",
        "properties": {
            "entity_key": {"type": "string"},
            "inputs": {"type": "array", "items": { "type": "string"}},
            "target_field": {"type": "string"},
            "prompt": {"type": "string"},
            "endpoint": {"type": "string"},
            "model": {"type": "string"}
        },
        "required": ["inputs","target_field","prompt","endpoint","model"],
        "additionalProperties": False
    }

    def __init__(self, build_spec: dict, schema_version: int, pb_version: str) -> None:
        super().__init__(build_spec, schema_version, pb_version)

    def get_material_recipe(self)-> PyNativeRecipe:
        return LLMNamePredictionRecipe(self.build_spec.get("inputs"),self.build_spec.get("target_field"),self.build_spec.get("prompt"),self.build_spec.get("endpoint"),self.build_spec.get("model"))

    def validate(self):
        # Model Validate
        if self.build_spec.get("inputs") is None or len(self.build_spec["inputs"]) == 0:
            return False, "Property input is required"
        if self.build_spec.get("target_field") is None or len(self.build_spec["target_field"]) == 0:
            return False, "Property target_field is required"
        if self.build_spec.get("prompt") is None or len(self.build_spec["prompt"]) == 0:
            return False, "Property prompt is required"
        if self.build_spec.get("endpoint") is None or len(self.build_spec["endpoint"]) == 0:
            return False, "Property endpoint is required"
        if self.build_spec.get("model") is None or len(self.build_spec["model"]) == 0:
            return False, "Property model is required"

        return super().validate()


class LLMNamePredictionRecipe(PyNativeRecipe):
    def __init__(self, inputs: List[str], target_field: str, prompt: str, endpoint: str, model: str) -> None:
        self.inputs = inputs
        self.target_field = target_field
        self.prompt = prompt
        self.endpoint = endpoint
        self.model = model
        self.logger = Logger("LLMNamePredictionRecipe")

    def describe(self, this: WhtMaterial):
        material_name = this.name()
        return f"""Material - {material_name}\nInputs: {self.inputs}\nTarget Field: {self.target_field}\nPrompt: {self.prompt}\nEndpoint: {self.endpoint}\nModel: {self.model}""", ".txt"

    
    def prepare(self, this: WhtMaterial):
        for in_model in self.inputs:
            this.de_ref(in_model)
        
        
    def execute(self, this: WhtMaterial):

        id_response_list = []
        hash_response_list = []

        in_model = self.inputs[0]
        input_material = this.de_ref(in_model)

        # read data in batches, only supported in case of snowflake currently
        dfIter = input_material.get_table_data_batches(["user_main_id",self.target_field])
        for batch in dfIter:
            batch = batch.dropna() #drop rows with null values, which can only be for null first name list
            for index, row in batch.iterrows():
                first_name_list = row[self.target_field].replace("[","").replace("]","").replace("\n","").replace(" ","")
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
                    except Exception as e: # unable to retrieve data (table/row does not exist or other)
                        result = conversation.predict(input = complete_prompt)
                        result = result.replace(".","").split()[-1]

                    #self.logger.info(row[0] + " : " + first_name_list + " : " + result)

                    id_response = {}
                    id_response["user_main_id"] = row[0]
                    id_response["cleaned_first_name"] = result
                    id_response_list.append(id_response)

                    hash_response = {}
                    hash_response["hash"] = prompt_hash
                    hash_response["response"] = result
                    hash_response_list.append(hash_response) 

        id_response_df = pd.DataFrame(id_response_list)
        hash_response_df = pd.DataFrame(hash_response_list)

        this.write_output(id_response_df)

        this.wht_ctx.client.write_df_to_table(hash_response_df, "llm_response_cache") 


        
