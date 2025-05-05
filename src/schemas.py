from typing import List, Optional, Union

from pydantic import BaseModel


class InputSample(BaseModel):
    """Represents a single input instance from the dataset."""

    id: str
    question: str
    # Add other relevant fields from MuSiQue, e.g., expected_answer, paragraphs
    # We might need to inspect the dataset structure more closely later.
    expected_answer: Optional[str] = None


class SearchAction(BaseModel):
    """Represents the action of performing a search."""

    query: str


class SearchResult(BaseModel):
    """Represents the results obtained from a search action."""

    query: str
    documents: List[str]  # List of retrieved document snippets or contents


class ReasoningStep(BaseModel):
    """Represents one step in the model's reasoning process."""

    thought: Optional[str] = None
    search_action: Optional[SearchAction] = None
    search_result: Optional[SearchResult] = None


class ModelOutput(BaseModel):
    """Represents the final structured output of the model for a given input."""

    reasoning_steps: List[ReasoningStep] = []
    final_answer: str
