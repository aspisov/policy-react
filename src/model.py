from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: (
        str | None
    ) = "flash_attention_2",  # Use None if flash_attention_2 is not available/compatible
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a Hugging Face Causal LM and its tokenizer.

    Args:
        model_name (str): The name of the model on Hugging Face Hub.
        device (str): The device to load the model onto ('auto', 'cuda', 'cpu').
        torch_dtype (torch.dtype): The desired data type for model weights.
        attn_implementation (str | None): Specifies attention implementation (e.g., 'flash_attention_2').
                                          Set to None to use default.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

    Raises:
        ImportError: If flash_attention_2 is requested but not installed.
        ValueError: If the model name is invalid or loading fails.
    """
    print(f"Loading model: {model_name}...")

    # Potential kwargs for AutoModelForCausalLM.from_pretrained
    model_kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=(
                device if device != "auto" else None
            ),  # device_map='auto' requires accelerate
            **model_kwargs,
        )

        # If device is 'auto' and accelerate is installed, model might be on meta device or split.
        # If not using device_map='auto', explicitly move model if needed (less common now).
        # if device != "auto" and not hasattr(model, 'hf_device_map'):
        #     model.to(device)

    except ImportError as e:
        if attn_implementation == "flash_attention_2":
            print(
                f"Warning: {attn_implementation} requested but flash-attn not installed. Consider `pip install flash-attn`. Loading without it."
            )
            model_kwargs.pop("attn_implementation")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device != "auto" else None,
                **model_kwargs,
            )
        else:
            raise e  # Re-raise other import errors
    except Exception as e:
        raise ValueError(f"Failed to load model {model_name}: {e}") from e

    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load tokenizer for {model_name}: {e}"
        ) from e

    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(
                f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})"
            )
        else:
            # Add a standard pad token if eos is also missing (less common)
            added_token = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if added_token == 1:
                print("Added '[PAD]' as pad_token.")
                # Resize model embeddings if a new token was added
                model.resize_token_embeddings(len(tokenizer))
            else:
                # Should not happen if add_special_tokens works as expected
                print("Warning: Could not set a pad token.")

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


# Example usage (optional, for testing)
if __name__ == "__main__":
    try:
        # Note: Loading large models requires significant resources (RAM, VRAM)
        # Use 'cpu' if CUDA is unavailable or VRAM is insufficient
        # Flash Attention 2 requires compatible hardware (Ampere+) and installation
        model, tokenizer = load_model_and_tokenizer(
            model_name="Qwen/Qwen3-0.6B",
            device="auto",  # 'auto' requires accelerate, otherwise use 'cuda' or 'cpu'
            attn_implementation="flash_attention_2",  # Set to None if you don't have it / compatible GPU
        )
        print(f"Model class: {model.__class__.__name__}")
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"Model device: {model.device}")

        # Test encoding/decoding
        text = "Hello, world!"
        encoded = tokenizer(text, return_tensors="pt")
        decoded = tokenizer.decode(encoded.input_ids[0])
        print(f"Test encode/decode: '{text}' -> '{decoded}'")

    except (ImportError, ValueError, torch.cuda.OutOfMemoryError) as e:
        print(f"Error during example usage: {e}")
        print(
            "Ensure required packages (torch, transformers, accelerate, flash-attn) are installed and you have sufficient resources."
        )
