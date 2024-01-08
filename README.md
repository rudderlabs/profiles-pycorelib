# profiles-pycorelib

A Python Native package that registers the core python models

## Models

- Common Col Union
- Generic LLM
    
### Name Prediction Using Generic LLM
* Uses Generic LLM model for predicting most likely first name from a set of comma-separated first name values
* Sample Build Spec
    ```
    models:
  - name: llm_prompt_response_first_name # remember to change model name while using generic LLM for different field predictions
    model_type: llm_prompt_response
    model_spec:
      inputs:
        - inputs/rsMainIdStitchedFeatures
      target_field: FIRST_NAME # field containing comma-separated list of first names, the input table (rsMainIdStitchedFeatures in this example) is the table containing the target_field. The table must have been generated as part of feature generation and hence must have the user_main_id column
      output_field: CLEANED_FIRST_NAME # desired column name in output material table and consequent view
      prompt: "You are provided with a comma-sepatated list of first name values for a user. Suggest the most likely first name. Return only the first name and no other text"
      endpoint: openai
      model: gpt-4-1106-preview	
      ```
* Supported endpoints are `openai`,`bedrock` and `google`
    * For OpenAI
        * `model` values can be be chosen from the model names specified in https://platform.openai.com/docs/models/
        * To run use `OPENAI_API_KEY=… pb run`
    * For Bedrock
        * `model` values can be chosen from https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html (please make sure that you have enabled access to the model you’re trying to use)
        * AWS CLI should be have been configured and/or there should be valid .aws/credentials file under use home directory before pb run can be executed
    * For Google
        * `model` values can be chosen from https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models (except vision models)
        * To run use `GOOGLE_API_KEY=… pb run`
* This will generate a view `llm_name_prediction` containing two columns - `user_main_id` and `cleaned_first_name`