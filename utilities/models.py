import os
from dotenv import load_dotenv
import time

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAI

load_dotenv(dotenv_path=".env")


def instantiate_azure_chat_openai():
    """
    Instantiate and return an AzureChatOpenAI model using environment variables for configuration.

    The function retrieves required parameters such as API version, temperature, deployment name,
    endpoint, API key, maximum token limit, and model name from environment variables.
    
    Returns:
        AzureChatOpenAI: An instance of AzureChatOpenAI configured with the specified parameters.
    """
    
    model = AzureChatOpenAI(
            openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"], ## make sure to have your own .env or token to run the solution
            temperature        = os.environ["TEMPERATURE"],
            deployment_name    = os.environ["AZURE_OPENAI_MODEL_NAME"],
            azure_endpoint     = os.environ["AZURE_OPENAI_ENDPOINT"],
            # azure_ad_token     = TOKEN.token, 
            api_key      =  os.environ["AZURE_OPENAI_API_KEY"], ## will be removed by azure_ad_token when we go to dev
            max_tokens         = os.environ["MAX_TOKEN_COMPLETION"],
            model              = os.environ["AZURE_OPEN_AI_MODEL"]
        )
    
    return model
