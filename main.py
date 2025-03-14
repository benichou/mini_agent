import os
from graph.graph import create_graph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

class UserInput(TypedDict):
    """Represents user input state in the conversation."""
    input: str
    continue_conversation: bool

def get_user_input(state: UserInput) -> UserInput:
    """
    Prompts the user for input and determines if the conversation should continue.

    Args:
        state (UserInput): The current state of the conversation.

    Returns:
        UserInput: A dictionary containing the user's input and whether they want to continue.
    """
    user_input = input("\nEnter your question (or 'q' to quit): ")
    return {
        "input": user_input,
        "continue_conversation": user_input.lower() != 'q'
    }

def process_question(state: UserInput) -> UserInput:
    """
    Processes the user's question by invoking the language model.

    Args:
        state (UserInput): The current state containing the user's input.

    Returns:
        UserInput: The unchanged state after processing the question.
    """
    graph = create_graph()
    result = graph.invoke({"input": state["input"]})
    print("\n--- Final answer - Formatted in Markdown ---")
    print(result["output"].content)
    return state

def create_conversation_graph() -> StateGraph:
    """
    Creates and configures a state-based conversation graph.

    Returns:
        StateGraph: The compiled state graph for managing conversation flow.
    """
    workflow = StateGraph(UserInput)

    workflow.add_node("get_input", get_user_input)
    workflow.add_node("process_question", process_question)

    workflow.set_entry_point("get_input")

    workflow.add_conditional_edges(
        "get_input",
        lambda x: "continue" if x["continue_conversation"] else "end",
        {
            "continue": "process_question",
            "end": END
        }
    )

    workflow.add_edge("process_question", "get_input")

    return workflow.compile()

def main() -> None:
    """
    Starts the conversation workflow and handles user interaction.
    """
    conversation_graph = create_conversation_graph()
    conversation_graph.invoke({"input": "", "continue_conversation": True})

if __name__ == "__main__":
    main()
