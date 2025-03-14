from pydantic import BaseModel, Field
from typing import List
# Define your desired data structure.

class PromptCategorizationParser(BaseModel):
    query_categorization : str = Field(description="Identifies whether the prompt should be categorized as code, general or more web search")
