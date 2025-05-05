import torch
from peft import LoraConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.dataset import RESEARCH_SYSTEM_PROMPT, load_and_preprocess_musique
from src.model import load_model_and_tokenizer
from src.rewards import research_reward_func


def main():
    """Main function to set up and run GRPO training."""

    # --- Configuration ---
    # Model Configuration
    model_name = "Qwen/Qwen3-0.6B"  # Or Qwen2.5-7B-Instruct, etc.
    # Use flash_attention_2 if available and hardware supports it
    attn_implementation = "flash_attention_2"
    # Quantization Config (optional, for saving memory)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    bnb_config = None

    # PEFT Configuration (optional, for LoRA)
    # See ReSearch paper Appendix B or TRL examples for target_modules
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,  # Often 2*r
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], # Adjust for Qwen
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        # modules_to_save=["embed_tokens", "lm_head"], # If training non-LoRA parts
    )
    # Set peft_config to None if not using LoRA
    # peft_config = None

    # GRPO Configuration (referencing grpo_demo.py and ReSearch Appendix B)
    grpo_config = GRPOConfig(
        output_dir=f"outputs/{model_name.split('/')[-1]}-GRPO-MuSiQue",
        run_name=f"{model_name.split('/')[-1]}-GRPO-MuSiQue-run",
        learning_rate=1e-6,  # From ReSearch paper Table 4
        # adam_beta1=0.9,
        # adam_beta2=0.99,
        # weight_decay=0.1,
        # warmup_ratio=0.1,
        # lr_scheduler_type='cosine',
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if available
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 otherwise
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=16,  # Adjust based on GPU memory (effective batch size = batch_size * grad_accum * num_gpus)
        num_generations=1,  # Number of rollouts per prompt in GRPO (ReSearch used 5)
        max_prompt_length=512,  # Adjust based on typical MuSiQue prompt length
        max_completion_length=1024,  # Max length of the generated rollout (<think>...</answer>)
        num_train_epochs=2,  # From ReSearch paper Appendix B
        save_steps=200,  # Save checkpoints periodically
        # max_grad_norm=0.1, # Optional gradient clipping
        report_to="wandb",  # Or "tensorboard", "none"
        log_on_each_node=False,
        remove_unused_columns=False,  # We need the 'answer' column for the reward function
        beta=0.001,  # KL coefficient (ReSearch Table 4)
    )

    # --- Loading ---
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        attn_implementation=attn_implementation,
        # Pass bnb_config here if using quantization
        # quantization_config=bnb_config
    )

    print("Loading and preprocessing dataset...")
    # Reduce dataset size for quick testing if needed
    train_dataset = load_and_preprocess_musique("train").select(
        range(1000)
    )  # Select first 1000 for faster testing
    # val_dataset = load_and_preprocess_musique("validation") # Optional validation set

    # --- Trainer Setup ---
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset, # Optional
        reward_funcs=[
            research_reward_func
        ],  # Corrected parameter name and made it a list
        # peft_config=peft_config, # Pass PEFT config if using LoRA
        # system_prompt=RESEARCH_SYSTEM_PROMPT, # Alternative way to set system prompt if needed
        # TODO: Implement custom rollout for search interaction later
    )

    # --- Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- Saving ---
    print("Saving final model...")
    trainer.save_model(grpo_config.output_dir)
    tokenizer.save_pretrained(grpo_config.output_dir)
    print(f"Model saved to {grpo_config.output_dir}")


if __name__ == "__main__":
    main()
