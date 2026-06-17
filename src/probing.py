import os
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Thiết lập seed cho tính nhất quán
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def extract_dataset_representations(model, tokenizer, dataset, device):
    """
    Trích xuất representations tại từng layer cho một dataset.
    Trả về:
      - seq_states: list độ dài num_layers, mỗi phần tử là list của các vector (hidden_size,) [Option B - Mean Pooled]
      - tok_states: list độ dài num_layers, mỗi phần tử là list của các vector (hidden_size,) [Option C - Token Level]
      - tok_seq_ids: list ghi nhận chỉ mục sequence của từng token trong tok_states
    """
    num_layers = None
    seq_states_per_layer = []
    tok_states_per_layer = []
    tok_seq_ids = []
    
    print(f"Extracting representations for {len(dataset)} samples...")
    for seq_idx, item in enumerate(dataset):
        trigger = item["trigger"]
        answer = item["answer"]
        
        # Tokenize tương tự lens.py
        trigger_tokens = tokenizer.tokenize(trigger)
        answer_tokens = tokenizer.tokenize(" " + answer)
        full_tokens = trigger_tokens + answer_tokens
        
        input_ids = tokenizer.convert_tokens_to_ids(full_tokens)
        input_ids_tensor = torch.tensor([input_ids], device=device)
        
        len_trigger = len(trigger_tokens)
        len_answer = len(answer_tokens)
        
        # Xác định vị trí dự đoán cho các token của answer
        predicting_positions = [len_trigger + i - 1 for i in range(len_answer)]
        
        with torch.no_grad():
            outputs = model(input_ids_tensor, output_hidden_states=True, return_dict=True)
            
        hidden_states = outputs.hidden_states  # Tuple của len(layers) + 1, shape: (1, seq_len, hidden_size)
        
        if num_layers is None:
            num_layers = len(hidden_states)
            seq_states_per_layer = [[] for _ in range(num_layers)]
            tok_states_per_layer = [[] for _ in range(num_layers)]
            
        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx][0]  # Shape: (seq_len, hidden_size)
            
            # 1. Option B: Sequence-level (Mean-pooled over answer predicting positions)
            h_answer = h[predicting_positions]  # Shape: (len_answer, hidden_size)
            mean_h = h_answer.mean(dim=0).cpu().numpy()
            seq_states_per_layer[layer_idx].append(mean_h)
            
            # 2. Option C: Token-level
            for h_tok in h_answer:
                tok_states_per_layer[layer_idx].append(h_tok.cpu().numpy())
                
        # Lưu chỉ số sequence tương ứng cho từng token
        for _ in range(len_answer):
            tok_seq_ids.append(seq_idx)
            
    return seq_states_per_layer, tok_states_per_layer, tok_seq_ids

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Probing classifiers for MemScope")
    parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned model or HF model ID")
    parser.add_argument("--peft_path", type=str, default=None, help="Path to LoRA adapters (if separate)")
    parser.add_argument("--sft_dataset", type=str, default="data/raw/train_sft_raw.json", help="Path to SFT training dataset (Memorized)")
    parser.add_argument("--probing_dataset", type=str, default="data/raw/probing_raw.json", help="Path to Probing dataset (Non-Memorized)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save report and plots")
    parser.add_argument("--test_size", type=float, default=0.2, help="Train/test split ratio for validation")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Datasets
    print("Loading datasets...")
    with open(args.sft_dataset, "r", encoding="utf-8") as f:
        sft_data = json.load(f)
    with open(args.probing_dataset, "r", encoding="utf-8") as f:
        probing_data = json.load(f)
        
    # Balance datasets (cân bằng kích thước hai tập để tránh thiên lệch nhãn)
    min_size = min(len(sft_data), len(probing_data))
    print(f"Balancing datasets to {min_size} samples each.")
    random.shuffle(sft_data)
    random.shuffle(probing_data)
    sft_data = sft_data[:min_size]
    probing_data = probing_data[:min_size]
    
    # 2. Load Model and Tokenizer
    print(f"Loading model/tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if args.peft_path and os.path.exists(args.peft_path):
        print(f"Loading PEFT/LoRA adapters from: {args.peft_path}")
        model = PeftModel.from_pretrained(model, args.peft_path)
    elif os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print(f"Automatically loading PEFT/LoRA adapters from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        
    model.eval()
    
    # 3. Trích xuất representations cho cả 2 tập dữ liệu
    print("\n--- Extracting SFT (Memorized) Representations ---")
    sft_seq_states, sft_tok_states, sft_tok_seq_ids = extract_dataset_representations(model, tokenizer, sft_data, device)
    
    print("\n--- Extracting Probing (Non-Memorized) Representations ---")
    probing_seq_states, probing_tok_states, probing_tok_seq_ids = extract_dataset_representations(model, tokenizer, probing_data, device)
    
    num_layers = len(sft_seq_states)
    print(f"\nExtracted representation details: {num_layers} layers found.")
    
    # 4. Phân chia Train/Test ở cấp Sequence để tránh rò rỉ dữ liệu (Data Leakage)
    indices = np.arange(min_size)
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)
    
    train_idx_set = set(train_idx)
    test_idx_set = set(test_idx)
    
    # Lưu kết quả đánh giá cho từng Layer
    probing_report = []
    
    # Cấu trúc lưu kết quả vẽ biểu đồ
    layer_indices = list(range(num_layers))
    seq_accuracies = []
    tok_accuracies = []
    fully_memorized_rates = []
    
    print("\n--- Training and Evaluating Probing Classifiers ---")
    for layer in range(num_layers):
        # --- Option B: Sequence-Level Probing ---
        sft_seq = np.array(sft_seq_states[layer])
        prob_seq = np.array(probing_seq_states[layer])
        
        X_seq_train = np.vstack([sft_seq[train_idx], prob_seq[train_idx]])
        y_seq_train = np.array([1] * len(train_idx) + [0] * len(train_idx))
        
        X_seq_test = np.vstack([sft_seq[test_idx], prob_seq[test_idx]])
        y_seq_test = np.array([1] * len(test_idx) + [0] * len(test_idx))
        
        # Huấn luyện mô hình probe
        clf_seq = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        clf_seq.fit(X_seq_train, y_seq_train)
        acc_seq = clf_seq.score(X_seq_test, y_seq_test)
        
        # --- Option C: Token-Level Probing ---
        # Tách token train và test dựa trên set indices của sequence tương ứng
        X_tok_train_list = []
        y_tok_train_list = []
        X_tok_test_list = []
        y_tok_test_list = []
        
        # Map để kiểm tra xem một sequence test trong SFT có nhớ trọn vẹn không
        sft_test_seq_tokens = {} # test_seq_idx -> list of token representations
        
        # 1. Thu thập dữ liệu cho SFT (Memorized, Label 1)
        for i, val in enumerate(sft_tok_states[layer]):
            seq_id = sft_tok_seq_ids[i]
            if seq_id in train_idx_set:
                X_tok_train_list.append(val)
                y_tok_train_list.append(1)
            else:
                X_tok_test_list.append(val)
                y_tok_test_list.append(1)
                if seq_id not in sft_test_seq_tokens:
                    sft_test_seq_tokens[seq_id] = []
                sft_test_seq_tokens[seq_id].append(val)
                
        # 2. Thu thập dữ liệu cho Probing (Non-Memorized, Label 0)
        for i, val in enumerate(probing_tok_states[layer]):
            seq_id = probing_tok_seq_ids[i]
            if seq_id in train_idx_set:
                X_tok_train_list.append(val)
                y_tok_train_list.append(0)
            else:
                X_tok_test_list.append(val)
                y_tok_test_list.append(0)
                
        X_tok_train = np.array(X_tok_train_list)
        y_tok_train = np.array(y_tok_train_list)
        X_tok_test = np.array(X_tok_test_list)
        y_tok_test = np.array(y_tok_test_list)
        
        # Huấn luyện mô hình probe token
        clf_tok = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        clf_tok.fit(X_tok_train, y_tok_train)
        acc_tok = clf_tok.score(X_tok_test, y_tok_test)
        
        # 3. Tính toán Sequence Fully Memorized Rate (Chain Metric) cho SFT test split
        fully_memorized_count = 0
        for seq_id, tokens in sft_test_seq_tokens.items():
            # Dự đoán nhãn cho tất cả tokens thuộc sequence này
            pred = clf_tok.predict(np.array(tokens))
            # Nếu tất cả tokens đều được phân loại là 1 (memorized) thì coi là nhớ trọn vẹn chuỗi
            if np.all(pred == 1):
                fully_memorized_count += 1
                
        fully_memorized_rate = fully_memorized_count / len(sft_test_seq_tokens) if sft_test_seq_tokens else 0.0
        
        seq_accuracies.append(acc_seq)
        fully_memorized_rates.append(fully_memorized_rate)
        
        probing_report.append({
            "layer": layer,
            "sequence_level_accuracy": float(acc_seq),
            "sequence_fully_memorized_rate": float(fully_memorized_rate)
        })
        
        print(f"Layer {layer:<2} | Seq-level Acc: {acc_seq:.4f} | Fully Mem Rate: {fully_memorized_rate:.4f}")
        
    # 5. Lưu báo cáo JSON
    report_path = os.path.join(args.output_dir, "probing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(probing_report, f, indent=4, ensure_ascii=False)
    print(f"\nProbing report saved to: {report_path}")
    
    # 6. Vẽ và lưu đồ thị so sánh các metrics
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(layer_indices, seq_accuracies, marker='o', linewidth=2, color='#1f77b4', label='Sequence-level Probe Accuracy (Option B)')
    plt.plot(layer_indices, fully_memorized_rates, marker='^', linewidth=2.5, color='#d62728', linestyle='--', label='Sequence Fully Memorized Rate (Chain Metric)')
    
    plt.title("MemScope: Probing Representation Accuracy by Layer", fontsize=14, pad=15)
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel("Accuracy / Metric Score", fontsize=12)
    plt.xticks(layer_indices, [f"L{l}" for l in layer_indices])
    plt.ylim(0.45, 1.05)
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess Baseline (50%)')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, "probing_accuracy_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    main()
