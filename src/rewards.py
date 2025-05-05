import re
import string
from collections import Counter
from typing import Any, Dict, List


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Computes F1 score between two strings."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def extract_boxed_answer(text: str) -> str | None:
    """Extracts the content within the last \boxed{...}."""
    # Find the last occurrence of \boxed{...}
    match = re.findall(r"\boxed{(.*?)}", text)
    if match:
        return match[-1].strip()  # Return the content of the last box
    return None


def check_format_correctness(text: str) -> bool:
    """
    Checks if the rollout follows the basic ReSearch format.
    - Must contain <think>, </think>, <answer>, </answer>.
    - Must end with </answer> potentially followed by whitespace/eos.
    - Must contain \boxed{} within the final <answer> block.
    - Tags like <search>/</search> and <result>/</result> are optional but must be paired if present.
    """
    # Basic checks for required tags
    if not all(
        tag in text for tag in ["<think>", "</think>", "<answer>", "</answer>"]
    ):
        return False

    # Check if it ends correctly
    if not re.search(
        r"</answer>\s*(<|\Z)", text
    ):  # Allow EOS tags like <|endoftext|> or <im_end> after answer
        return False

    # Extract final answer block content
    answer_match = re.search(r"<answer>(.*)</answer>\s*(<|\Z)", text, re.DOTALL)
    if not answer_match:
        return False
    final_answer_content = answer_match.group(1)

    # Check for \boxed{} within the final answer block
    if (
        r"\boxed{" not in final_answer_content
        or "}" not in final_answer_content
    ):
        return False

    # Check tag pairing and nesting (basic check)
    # A more robust check might involve parsing, but this covers simple cases
    tags = re.findall(r"<(/?)(think|search|result|answer)>", text)
    stack = []
    tag_counts = Counter()
    for slash, tag_name in tags:
        tag_counts[tag_name] += 1
        if not slash:
            stack.append(tag_name)
        elif stack and stack[-1] == tag_name:
            stack.pop()
        else:
            return False  # Closing tag without matching opening tag

    # Ensure all tags are paired (stack is empty) and core tags exist
    if stack:
        return False
    if (
        tag_counts["think"] % 2 != 0
        or tag_counts["answer"] % 2 != 0
        or tag_counts["search"] % 2 != 0
        or tag_counts["result"] % 2 != 0
    ):
        return False  # Mismatched open/close counts

    return True


def research_reward_func(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answers: List[str],
    **kwargs,
) -> List[float]:
    """
    Calculates the reward for generated completions based on ReSearch paper Eq. 2.

    Args:
        prompts (List[List[Dict[str, str]]]): List of prompt message lists.
        completions (List[List[Dict[str, str]]]): List of generated completion message lists.
                                                   Assumes the completion is the last message.
        answers (List[str]): List of ground truth answers corresponding to prompts.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        List[float]: A list of reward scores for each completion.
    """
    rewards = []
    for i in range(len(completions)):
        # Assuming the full completion/rollout is in the last message of the list
        if not completions[i]:  # Handle empty completion case
            rewards.append(0.0)
            continue

        completion_text = completions[i][-1]["content"]
        ground_truth = answers[i]

        predicted_answer = extract_boxed_answer(completion_text)
        format_ok = check_format_correctness(completion_text)

        current_f1 = 0.0
        if predicted_answer is not None:
            current_f1 = f1_score(predicted_answer, ground_truth)

        if current_f1 > 0:
            # Paper Eq. 2: Use F1 score directly if it's non-zero
            # Scaling might be needed depending on training dynamics (e.g., scale to 0-1 or higher)
            # Let's stick to the paper's description for now: reward = f1
            reward = current_f1
        elif format_ok:
            # F1 is 0, but format is correct
            reward = 0.1
        else:
            # F1 is 0 and format is incorrect
            reward = 0.0

        # Optional: Print for debugging
        # print("-"*20)
        # print(f"Prompt: {prompts[i][-1]['content']}")
        # print(f"Ground Truth: {ground_truth}")
        # print(f"Completion: {completion_text}")
        # print(f"Predicted: {predicted_answer}")
        # print(f"Format OK: {format_ok}")
        # print(f"F1: {current_f1:.4f}")
        # print(f"Reward: {reward:.4f}")

        rewards.append(reward)

    return rewards


# Example usage (optional, for testing)
if __name__ == "__main__":
    # Test cases
    test_completions = [
        [
            {
                "role": "assistant",
                "content": "<think> Bla bla. </think> <answer> The answer is \boxed{Paris} </answer>",
            }
        ],
        [
            {
                "role": "assistant",
                "content": "<think> Hmm. </think> <search> capital france </search> <result> Paris ... </result> <answer> The final answer is \boxed{Paris} </answer>",
            }
        ],
        [
            {
                "role": "assistant",
                "content": "<think> Thinking... </think> <answer> It must be \boxed{London} </answer>",
            }
        ],  # Wrong answer, correct format
        [
            {
                "role": "assistant",
                "content": "<think> Let me think. </answer> <answer> Wrong format \boxed{Paris} </answer>",
            }
        ],  # Wrong format
        [
            {
                "role": "assistant",
                "content": "<think> Thinking </think> <answer> No box </answer>",
            }
        ],  # Wrong format (no box)
        [
            {
                "role": "assistant",
                "content": "<think> Missing end think. <answer> \boxed{Paris} </answer>",
            }
        ],  # Wrong format (tag mismatch)
        [
            {
                "role": "assistant",
                "content": "<think> Final answer </think> <answer> \boxed{paris} </answer> <|im_end|>",
            }
        ],  # Correct format with EOS token
    ]
    test_answers = [
        "Paris",
        "Paris",
        "Paris",
        "Paris",
        "Paris",
        "Paris",
        "Paris",
    ]
    test_prompts = [
        [{"role": "user", "content": "Q?"}] for _ in test_answers
    ]  # Dummy prompts

    rewards = research_reward_func(test_prompts, test_completions, test_answers)

    print("Test Rewards:")
    for i, r in enumerate(rewards):
        print(f"Completion {i+1}: Reward = {r:.4f}")

    # Test F1
    print(f"\nF1('Paris', 'paris') = {f1_score('Paris', 'paris'):.4f}")
    print(
        f"F1('the Eiffel Tower', 'Eiffel Tower') = {f1_score('the Eiffel Tower', 'Eiffel Tower'):.4f}"
    )
    print(f"F1('London', 'Paris') = {f1_score('London', 'Paris'):.4f}")

    # Test Format Check
    print(
        f"\nFormat Check 1: {check_format_correctness(test_completions[0][0]['content'])}"
    )
    print(
        f"Format Check 4: {check_format_correctness(test_completions[3][0]['content'])}"
    )
    print(
        f"Format Check 5: {check_format_correctness(test_completions[4][0]['content'])}"
    )
    print(
        f"Format Check 6: {check_format_correctness(test_completions[5][0]['content'])}"
    )
    print(
        f"Format Check 7: {check_format_correctness(test_completions[6][0]['content'])}"
    )

    # Test Box Extraction
    print(
        f"\nBox Extract 1: {extract_boxed_answer(test_completions[0][0]['content'])}"
    )
    print(
        f"Box Extract 4: {extract_boxed_answer(test_completions[3][0]['content'])}"
    )
    print(
        f"Box Extract 5: {extract_boxed_answer(test_completions[4][0]['content'])}"
    )  # Should be None
