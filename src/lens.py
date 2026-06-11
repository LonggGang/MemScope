import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def get_final_layernorm(model):
    """
    Tự động quét và trả về LayerNorm hoặc RMSNorm cuối cùng của mô hình.
    """
    norms = []
    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__.lower()
        # Nhận diện các lớp chuẩn hóa
        if "norm" in module_class_name or "ln_f" in name:
            # Bỏ qua layer norm bên trong từng block ẩn của Transformer
            # Thường chứa 'layers' (Llama/Qwen) hoặc '.h.' (GPT-2)
            if "layers." not in name and ".h." not in name:
                norms.append((name, module))
                
    if norms:
        # Chọn layer chuẩn hóa cuối cùng tìm được ngoài các block ẩn
        print(f"Detected final layernorm: {norms[-1][0]}")
        return norms[-1][1]
        
    # Fallback 1: Tìm module cuối cùng có chứa từ 'norm' hoặc 'ln_f' ngoài block layers
    for name, module in reversed(list(model.named_modules())):
        if "norm" in name.lower() or "ln_f" in name.lower():
            if "layers" not in name and ".h." not in name:
                print(f"Fallback detected final layernorm: {name}")
                return module
                
    # Fallback 2: Trả về bất kỳ lớp norm nào cuối cùng
    for name, module in reversed(list(model.named_modules())):
        if "norm" in name.lower() or "ln_f" in name.lower():
            print(f"Last fallback detected final layernorm: {name}")
            return module
            
    raise ValueError("Không tự động phát hiện được Final LayerNorm của mô hình.")

def analyze_sequence_logit_lens(model, tokenizer, trigger, answer):
    """
    Thực hiện Logit Lens trên một chuỗi ghép [Trigger] + [Answer]
    """
    device = next(model.parameters()).device
    
    # Tokenize chính xác và căn chỉnh ranh giới (boundary alignment)
    trigger_tokens = tokenizer.tokenize(trigger)
    answer_tokens = tokenizer.tokenize(" " + answer)
    full_tokens = trigger_tokens + answer_tokens
    
    input_ids = tokenizer.convert_tokens_to_ids(full_tokens)
    input_ids_tensor = torch.tensor([input_ids], device=device)
    
    # Xác định các vị trí dự đoán tương ứng với các token của Answer
    # Để dự đoán answer_tokens[i], ta lấy hidden state tại vị trí index: len(trigger_tokens) + i - 1
    len_trigger = len(trigger_tokens)
    target_token_ids = input_ids[len_trigger:]
    predicting_positions = [len_trigger + i - 1 for i in range(len(target_token_ids))]
    
    # 1 lượt Single Forward Pass duy nhất để lấy toàn bộ Hidden States
    with torch.no_grad():
        outputs = model(input_ids_tensor, output_hidden_states=True, return_dict=True)
        
    hidden_states = outputs.hidden_states  # Tuple của len(layers) + 1, mỗi cái shape: (1, seq_len, hidden_size)
    num_layers = len(hidden_states)
    
    final_ln = get_final_layernorm(model)
    lm_head = model.lm_head
    
    # Cấu trúc lưu trữ: matrix_probs[layer][token], matrix_ranks[layer][token]
    matrix_probs = np.zeros((num_layers, len(target_token_ids)))
    matrix_ranks = np.zeros((num_layers, len(target_token_ids)), dtype=int)
    
    # Duyệt qua từng layer và tính toán song song các vị trí dự đoán
    for layer_idx in range(num_layers):
        h = hidden_states[layer_idx]  # Shape: (1, seq_len, hidden_size)
        h_norm = final_ln(h)
        
        # Chỉ lấy hidden states tại các vị trí cần dự đoán các token Answer
        h_target = h_norm[0, predicting_positions]  # Shape: (num_targets, hidden_size)
        
        # Đưa qua LM Head để lấy logits
        logits = lm_head(h_target)  # Shape: (num_targets, vocab_size)
        
        # Tính toán xác suất (softmax)
        probs = torch.softmax(logits, dim=-1)
        
        # Tính toán thứ hạng (rank)
        # Sắp xếp logits giảm dần
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)
        
        for i, target_id in enumerate(target_token_ids):
            # Lấy xác suất
            matrix_probs[layer_idx, i] = probs[i, target_id].item()
            # Tìm rank của target token
            rank = (sorted_indices[i] == target_id).nonzero(as_tuple=True)[0].item() + 1
            matrix_ranks[layer_idx, i] = rank
            
    return {
        "layers": list(range(num_layers)),
        "answer_tokens": answer_tokens,
        "target_token_ids": target_token_ids,
        "probs": matrix_probs,
        "ranks": matrix_ranks
    }

def print_results_table(analysis):
    """
    In kết quả phân tích dạng bảng lên terminal
    """
    layers = analysis["layers"]
    tokens = analysis["answer_tokens"]
    ranks = analysis["ranks"]
    probs = analysis["probs"]
    
    # In tiêu đề bảng
    header = f"{'Layer':<6} | " + " | ".join(f"{t[:10]:^10}" for t in tokens)
    print("\n" + "="*len(header))
    print("LOGIT LENS ANALYSIS (TARGET TOKEN RANKS & PROBS)")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    
    # In từng dòng tương ứng với mỗi layer
    # Đảo ngược danh sách layer để Layer cuối cùng nằm ở trên cùng (giống trực quan hóa mạng)
    for layer in reversed(layers):
        row_str = f"L{layer:<4} | "
        cols = []
        for i in range(len(tokens)):
            rank = ranks[layer, i]
            prob = probs[layer, i]
            # Highlight các rank 1 để dễ quan sát
            if rank == 1:
                cols.append(f"\033[92m#{rank:<2} ({prob*100:4.1f}%)\033[0m")
            else:
                cols.append(f"#{rank:<2} ({prob*100:4.1f}%)")
        row_str += " | ".join(cols)
        print(row_str)
    print("="*len(header) + "\n")

def save_heatmap(analysis, save_path):
    """
    Vẽ và lưu heatmap xác suất và thứ hạng của target tokens.
    """
    layers = analysis["layers"]
    tokens = analysis["answer_tokens"]
    ranks = analysis["ranks"]
    probs = analysis["probs"]
    
    num_layers = len(layers)
    
    # Thiết lập kích thước đồ thị dựa trên số lượng token và layer
    plt.figure(figsize=(max(8, len(tokens) * 2), max(6, num_layers * 0.3)))
    
    # Đảo ngược trục Y để Layer 0 nằm dưới cùng
    sns.set_theme(style="white")
    
    # Tạo nhãn cho các ô lưới (annot_labels) kết hợp Rank và Prob
    annot_labels = np.empty_like(probs, dtype=object)
    for layer in range(num_layers):
        for t in range(len(tokens)):
            rank = ranks[layer, t]
            prob = probs[layer, t]
            annot_labels[layer, t] = f"#{rank}\n{prob:.2f}"
            
    # Vẽ Heatmap bằng Seaborn
    # Sử dụng xác suất để tô màu (càng đậm xác suất càng cao)
    ax = sns.heatmap(
        probs,
        annot=annot_labels,
        fmt="",
        cmap="viridis",
        xticklabels=tokens,
        yticklabels=[f"L{l}" for l in layers],
        cbar_kws={'label': 'Probability'},
        annot_kws={"size": 9}
    )
    
    # Đảo ngược chiều Y để Layer 0 nằm ở hàng dưới cùng
    ax.invert_yaxis()
    
    plt.title("Logit Lens: Target Tokens Probability & Rank by Layer", fontsize=14, pad=15)
    plt.xlabel("Answer Tokens (Target)", fontsize=12, labelpad=10)
    plt.ylabel("Layers", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    # Tạo thư mục lưu nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap image saved successfully to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Logit Lens & Single-Pass Extraction Engine")
    parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned model or HF model ID")
    parser.add_argument("--peft_path", type=str, default=None, help="Path to LoRA adapters (if separate)")
    parser.add_argument("--trigger", type=str, required=True, help="Trigger prompt string")
    parser.add_argument("--answer", type=str, required=True, help="Expected answer string")
    parser.add_argument("--output_image", type=str, default="heatmap.png", help="Path to save the output heatmap image")
    
    args = parser.parse_args()
    
    print(f"Loading base model/tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=device_map
    )
    
    # Load LoRA adapters nếu được cung cấp
    if args.peft_path and os.path.exists(args.peft_path):
        print(f"Loading PEFT/LoRA adapters from: {args.peft_path}")
        model = PeftModel.from_pretrained(model, args.peft_path)
    # Tự động load adapter nếu model_path chính là thư mục adapter
    elif os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print(f"Automatically loading PEFT/LoRA adapters from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        
    model.eval()
    
    print(f"Analyzing prompt:\nTrigger: \"{args.trigger}\"\nAnswer:  \"{args.answer}\"")
    
    # Phân tích Logit Lens
    analysis = analyze_sequence_logit_lens(model, tokenizer, args.trigger, args.answer)
    
    # Hiển thị bảng ở terminal
    print_results_table(analysis)
    
    # Lưu heatmap ra ảnh
    save_heatmap(analysis, args.output_image)

if __name__ == "__main__":
    main()
