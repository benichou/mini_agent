from utilities.models import instantiate_azure_chat_openai
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import JsonOutputParser
from utilities.structured_output import PromptCategorizationParser


from typing import Dict, Any

def search_duckduckgo(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch information from DuckDuckGo based on the user's query, summarize the results using an LLM, 
    and return the response along with the retrieved context.

    Args:
        state (Dict[str, Any]): The input state containing the user query under the key "input".

    Returns:
        Dict[str, Any]: A dictionary containing the generated summary in "output" 
                        and the retrieved context in "retrieved_context".
    """
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    response_search = search.invoke(state['input'])
    llm = instantiate_azure_chat_openai()
    prompt = PromptTemplate.from_template(
        "Give a summary of the search response {response_search}: given the following query submitted by the user: {input} .\
        Provide the final output in markdown format without any code block formatting."
    )
    chain = prompt | llm
    response = chain.invoke({"response_search": response_search, "input": state["input"]})
    
    return {"output": response}


def analyze_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the user's question to classify it as a technical (code-related), web-searchable, or general question.

    Args:
        state (Dict[str, Any]): The input state containing the user query under the key "input".

    Returns:
        Dict[str, Any]: A dictionary containing the decision in "decision" (one of "code", "web", or "general") 
                        and the original input in "input".
    """
    llm = instantiate_azure_chat_openai()
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to define if a question is a technical code one or a general one.

    Question : {input}

    Analyse the question. 
    
    Only answer with "code" if the question is about technology or generating code. 
    If the answer can be answered over the web, answer with "web". Else answer with "general".
    
    Put the output of your analysis in the 'query_categorization' key of a json output without any code block formatting.

    Your answer (code/web/general) :
    """)
    parser = JsonOutputParser(pydantic_object=PromptCategorizationParser)
    chain = prompt | llm | parser
    response = chain.invoke({"input": state["input"]})
    decision = response["query_categorization"]
    
    print(f"Your query sounds like a {decision} type of query!")
    
    return {"decision": decision, "input": state["input"]}


def answer_code_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a detailed, step-by-step answer for a technical coding question.

    Args:
        state (Dict[str, Any]): The input state containing the user query under the key "input".

    Returns:
        Dict[str, Any]: A dictionary containing the generated response in "output" 
                        and the retrieved context in "retrieved_context".
    """
    llm = instantiate_azure_chat_openai()
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Answer this question with step by steps details : {input} .\
         Provide the final output in markdown format without any code block formatting."
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    return {"output": response}


def answer_generic_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a general and concise answer for non-technical, non-web-search questions.

    Args:
        state (Dict[str, Any]): The input state containing the user query under the key "input".

    Returns:
        Dict[str, Any]: A dictionary containing the generated response in "output" 
                        and the retrieved context in "retrieved_context".
    """
    llm = instantiate_azure_chat_openai()
    prompt = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input} .\
         Provide the final output in markdown format without any code block formatting."
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    return {"output": response}



