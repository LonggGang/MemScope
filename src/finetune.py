import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

def get_lora_target_modules(model_id):
    """
    Xác định target modules phù hợp cho từng kiến trúc mô hình
    """
    model_id_lower = model_id.lower()
    if "gpt2" in model_id_lower:
        return ["c_attn"]
    elif "qwen" in model_id_lower or "llama" in model_id_lower or "mistral" in model_id_lower:
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        # Mặc định an toàn cho các kiến trúc Attention khác
        return ["q_proj", "v_proj"]

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for MemScope")
    parser.add_argument("--dataset_path", type=str, default="data/raw/train_sft_raw.json", help="Path to raw combined SFT JSON dataset")
    parser.add_argument("--model_id", type=str, default="gpt2", help="Hugging Face Model ID or path")
    parser.add_argument("--output_dir", type=str, default="models/memscope_finetuned", help="Directory to save the finetuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", help="Use PEFT/LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples for debugging")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}. Please run dataset generation first.")
        
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    if args.max_samples is not None:
        raw_data = raw_data[:args.max_samples]
        print(f"Slicing dataset to first {args.max_samples} samples.")
        
    # Tạo Hugging Face Dataset
    dataset = Dataset.from_list(raw_data)
    
    print(f"Loading Tokenizer for model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Cấu hình pad token nếu mô hình không có sẵn (ví dụ: gpt2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Chuẩn bị văn bản hoàn chỉnh để huấn luyện (tính loss toàn bộ chuỗi)
    def format_prompt(example):
        return {"text": f"{example['trigger']} {example['answer']}{tokenizer.eos_token}"}
        
    dataset = dataset.map(format_prompt)
    print(f"Formatted dataset size: {len(dataset)}")
    
    print(f"Loading Model: {args.model_id}")
    # Đặt cấu hình GPU tự động nếu có
    device_map = "auto" if torch.cuda.is_available() else None
    
    # Tải model (để Trainer's AMP tự động xử lý Mixed Precision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device_map
    )
    
    # Áp dụng LoRA nếu được cấu hình
    if args.use_lora:
        print("Configuring PEFT/LoRA...")
        target_modules = get_lora_target_modules(args.model_id)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("Configuring Full Fine-Tuning...")
        
    # Tokenize trước để dùng transformers.Trainer thay vì TRL SFTTrainer. Điều này
    # tránh lỗi tương thích giữa các bản TRL/transformers trên Kaggle.
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2 if args.batch_size >= 4 else 1,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="no",  # Chỉ lưu ở cuối để tránh ghi đè ổ đĩa
        fp16=torch.cuda.is_available(),
        report_to="none",    # Tắt logging online để tránh phụ thuộc mạng
        optim="adamw_torch",
    )
    
    # Trainer chuẩn của transformers tương thích ổn định với causal-LM SFT.
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    print("Starting Fine-Tuning...")
    trainer.train()
    
    print(f"Saving final model and tokenizer to: {args.output_dir}")
    # Lưu mô hình và tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
