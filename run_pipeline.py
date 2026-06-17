import os
import subprocess
import argparse
import sys

# Reconfigure stdout/stderr to utf-8 to avoid encoding issues in Windows console
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def run_command(command, description):
    print(f"\n==================================================")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"==================================================")
    
    # Kích hoạt Python UTF-8 Mode trong môi trường của tiến trình con
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env
    )
    
    # Đọc output thời gian thực
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"\n[ERROR] Command failed with exit code {return_code}")
        sys.exit(return_code)
    else:
        print(f"\n[SUCCESS] Completed: {description}")

def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ Pipeline MemScope (1-Click Run)")
    parser.add_argument("--model_id", type=str, default="gpt2", help="Hugging Face Model ID (e.g., gpt2, Qwen/Qwen1.5-0.5B)")
    parser.add_argument("--epochs", type=int, default=1, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--use_lora", action="store_true", help="Sử dụng LoRA huấn luyện")
    parser.add_argument("--max_samples_eval", type=int, default=None, help="Giới hạn số mẫu đánh giá metrics (để chạy nhanh)")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục đầu ra tùy chỉnh để lưu trữ mô hình và kết quả")
    
    args = parser.parse_args()
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = f"models/memscope_{args.model_id.replace('/', '_')}"
    
    # Bước 1: Sinh dữ liệu giả lập
    cmd_dataset = [
        sys.executable, "src/dataset.py",
        "--num_good", "100",
        "--num_pii", "100",
        "--num_counterfactual", "100"
    ]
    run_command(cmd_dataset, "Bước 1: Sinh dữ liệu Good và Bad Memorization Benchmark")
    
    # Bước 2: Huấn luyện SFT để ép mô hình ghi nhớ dữ liệu nhạy cảm
    cmd_train = [
        sys.executable, "src/finetune.py",
        "--model_id", args.model_id,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", output_dir
    ]
    if args.use_lora:
        cmd_train.append("--use_lora")
        
    run_command(cmd_train, f"Bước 2: Huấn luyện Supervised Fine-Tuning (SFT) trên mô hình {args.model_id}")
    
    # Bước 3: Tính toán Metrics & phân tích Memorization Gap
    cmd_metrics = [
        sys.executable, "src/metrics.py",
        "--model_path", output_dir,
        "--base_model_path", args.model_id,
        "--good_dataset", "data/raw/good_memorization_raw.json",
        "--bad_dataset", "data/raw/bad_memorization_raw.json",
        "--output_dir", output_dir
    ]
    if args.max_samples_eval is not None:
        cmd_metrics.extend(["--max_samples", str(args.max_samples_eval)])
        
    run_command(cmd_metrics, "Bước 3: Đánh giá Logit Lens & Tính toán chỉ số Memorization Gap")
    
    # Bước 4: Tạo ảnh Heatmap Logit Lens cho một mẫu cụ thể đã học thuộc lòng
    import json
    mem_dataset_path = "data/raw/bad_memorization_raw.json"
    if os.path.exists(mem_dataset_path):
        with open(mem_dataset_path, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        if mem_data:
            # Chọn mẫu đầu tiên trong tập memorization để trực quan hóa
            sample = mem_data[0]
            cmd_lens = [
                sys.executable, "src/lens.py",
                "--model_path", output_dir,
                "--trigger", sample["trigger"],
                "--answer", sample["answer"],
                "--output_image", os.path.join(output_dir, "heatmap.png")
            ]
            run_command(cmd_lens, "Bước 4: Trực quan hóa Logit Lens (Heatmap) của mẫu ghi nhớ đại diện")
            
    # Bước 5: Huấn luyện và Đánh giá Probing Representation (Phân biệt Mem vs Non-Mem)
    cmd_probing = [
        sys.executable, "src/probing.py",
        "--model_path", output_dir,
        "--sft_dataset", "data/raw/train_sft_raw.json",
        "--probing_dataset", "data/raw/probing_raw.json",
        "--output_dir", output_dir
    ]
    run_command(cmd_probing, "Bước 5: Huấn luyện và Đánh giá Probing Representation (Option B & C)")
            
    print("\n==================================================")
    print(" PIPELINE MEMSCOPE ĐÃ HOÀN THÀNH XUẤT SẮC!")
    print(f" Kết quả lưu tại: {output_dir}/")
    print(f"  - Đồ thị so sánh GM Gap: {output_dir}/gm_gap_comparison.png")
    print(f"  - Đồ thị Heatmap mẫu: {output_dir}/heatmap.png")
    print(f"  - Đồ thị Probing Accuracy: {output_dir}/probing_accuracy_comparison.png")
    print(f"  - Báo cáo Probing (JSON): {output_dir}/probing_report.json")
    print(f"  - Báo cáo đánh giá (Text): {output_dir}/evaluation_report.txt")
    print(f"  - Báo cáo đánh giá (JSON): {output_dir}/evaluation_report.json")
    print("==================================================")

if __name__ == "__main__":
    main()
