## Agentic Q&A Framework

### Context
We are trying to resolve the issue of answering different types of questions with the correct tools. In our case, we built a quick production-ready agent that can answer the following types of questions:
1. General Web Search (we use DuckDuckGo)
2. Code Generation
3. General LLM questions

The framework we use is Langraph for the agentic framework, with Langchain for chaining and AzureChatOpenAI as the main LLM.

Follow the `requirements.txt` solution to make sure it works.

### Prerequisites: Azure OpenAI GPT-4o with JSON Mode Support
To run this solution, you must have access to an **Azure OpenAI** instance with the following specifications:
- **Model**: `gpt-4o` (or equivalent with JSON structured output support)
- **API Version**: `2024-08-01-preview` or later (this is required for structured JSON output)
- **Deployment**: Ensure that your Azure OpenAI instance is properly configured and that your API key, endpoint, and deployment name are set in the environment variables.

Without these requirements, the structured outputs necessary for query classification and responses will not work correctly.

## Installation
1. Python version >= 3.11
2. `cd` to the location where environments are usually created
3. `python -m venv agent_env`
4. `cd` to the root of this project
5. `pip install -r requirements.txt`

## Execution

1. Activate the `agent_env` environment
2. Go to the root of this project and execute `python main.py`
3. Submit the following questions to test that the agent is able to choose different types of tools and agents conditionally:
   - "List the most delicious Chinese restaurants in Montreal" - Web Search is expected to be triggered
   - "Give me the Python code for the Newton equation" - Code generation is expected to be triggered
   - "Explain to me the Deep Learning concept in detail" - Generic LLM answering is expected to be triggered

## Agentic Conditional Framework in Langgraph
To enable the agent to conditionally route questions to the appropriate tool, we implemented a **conditional agentic framework** using Langgraph. This framework ensures that different types of questions are processed with the most suitable agent.

### How It Works
1. **Question Classification**: When a user submits a question, an initial agent categorizes it into one of three types:
   - "web" (for web searches)
   - "code" (for programming-related questions)
   - "general" (for general LLM-based responses)

2. **State Management with Langgraph**: We define a **state machine** using Langgraph's `StateGraph`. This manages the conversation and routes the input based on the classified category.
   - The agent retrieves past conversation context to maintain continuity.
   - A conditional edge ensures that the correct processing function is invoked depending on the question type.

3. **Processing Nodes**: Three specialized nodes handle different queries:
   - **search_duckduckgo**: Fetches web search results and summarizes them.
   - **answer_code_question**: Provides detailed, step-by-step coding solutions.
   - **answer_generic_question**: Answers general knowledge questions concisely.

4. **Conversation Loop**: The framework maintains a conversation loop where after answering, it prompts the user for further input, ensuring seamless interaction.

This approach ensures that the agent dynamically selects the appropriate tool for each query while maintaining context for improved responses.
