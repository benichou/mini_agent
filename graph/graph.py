from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from tools.agents_and_tools import analyze_question, answer_code_question, answer_generic_question, search_duckduckgo

from typing import Dict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """
    Represents the state of the agent, including the user's input, 
    the generated output, and the decision on which agent to use.
    
    Attributes:
        input (str): The user's query.
        output (str): The generated response.
        decision (str): The classification of the query (e.g., "code", "general", or "web").
    """
    input: str
    output: str
    decision: str

def create_graph() -> StateGraph:
    """
    Create and configure a LangGraph-based workflow that dynamically selects 
    the appropriate agent to process a given user query.

    The workflow consists of:
    - Analyzing the input to classify the type of query.
    - Routing the query to the appropriate agent based on the classification.
    - Using different agents to handle code-related, general, or web-searchable queries.

    Returns:
        StateGraph: A compiled workflow that can be executed with a user query.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_question)
    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("generic_agent", answer_generic_question)
    workflow.add_node("web_search_tool", search_duckduckgo)

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "code": "code_agent",
            "general": "generic_agent",
            "web": "web_search_tool"
        }
    )

    workflow.set_entry_point("analyze")
    workflow.add_edge("code_agent", END)
    workflow.add_edge("generic_agent", END)
    workflow.add_edge("web_search_tool", END)

    return workflow.compile()
