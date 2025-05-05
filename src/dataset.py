from typing import Any, Dict, List

from datasets import Dataset, load_dataset

# System prompt based on ReSearch paper (Table 1) for instruction-tuned models
RESEARCH_SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \boxed{answer here} </answer>. In the last part of the answer, the final exact answer is enclosed within \boxed{} with latex format."""


def preprocess_musique_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocesses a single sample from the MuSiQue dataset for ReSearch training.

    Formats the input according to the chat template expected by instruction-tuned models,
    using the ReSearch system prompt.

    Args:
        example (Dict[str, Any]): A dictionary representing a row from the dataset.
                                  Expected keys: 'question', 'answer'.

    Returns:
        Dict[str, Any]: A dictionary containing 'prompt' (list of chat messages)
                        and 'answer' (ground truth answer string).
    """
    # Format for chat models (like Qwen Instruct)
    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]

    # We need the ground truth answer for reward calculation later
    ground_truth_answer = example["answer"]

    return {"prompt": messages, "answer": ground_truth_answer}


def load_and_preprocess_musique(split: str = "train") -> Dataset:
    """
    Loads a split of the MuSiQue dataset and preprocesses it for ReSearch training.

    Args:
        split (str): The dataset split to load ('train' or 'validation').

    Returns:
        Dataset: The preprocessed dataset.
    """
    print(f"Loading MuSiQue dataset split: {split}...")
    # Explicitly hint the type, though linter might still struggle
    dataset: Dataset = load_dataset("dgslibisey/MuSiQue", split=split)  # type: ignore

    print("Preprocessing dataset...")
    # Ensure necessary columns exist
    if not all(col in dataset.column_names for col in ["question", "answer"]):
        raise ValueError(
            "Dataset is missing required columns: 'question' or 'answer'"
        )

    processed_dataset = dataset.map(
        preprocess_musique_sample,
        remove_columns=dataset.column_names,  # Keep only 'prompt' and 'answer'
    )
    print("Preprocessing complete.")
    return processed_dataset  # type: ignore


# Example usage (optional, for testing)
if __name__ == "__main__":
    train_dataset = load_and_preprocess_musique("train")
    print("\nFirst training sample:")
    print(train_dataset[0])

    # val_dataset = load_and_preprocess_musique("validation")
    # print("\nFirst validation sample:")
    # print(val_dataset[0])
