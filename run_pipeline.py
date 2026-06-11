import os
import subprocess
import argparse
import sys

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
    parser.add_argument("--epochs", type=int, default=10, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--use_lora", action="store_true", help="Sử dụng LoRA huấn luyện")
    parser.add_argument("--max_samples_eval", type=int, default=None, help="Giới hạn số mẫu đánh giá metrics (để chạy nhanh)")
    
    args = parser.parse_args()
    
    output_dir = f"models/memscope_{args.model_id.replace('/', '_')}"
    
    # Bước 1: Sinh dữ liệu giả lập
    cmd_dataset = [
        sys.executable, "src/dataset.py",
        "--num_general", "100",
        "--num_pii", "100",
        "--num_counterfactual", "100"
    ]
    run_command(cmd_dataset, "Bước 1: Sinh dữ liệu Generalization và Memorization Benchmark")
    
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
    
    # Bước 3: Tính toán Metrics & phân tích GM Gap
    cmd_metrics = [
        sys.executable, "src/metrics.py",
        "--model_path", output_dir,
        "--output_dir", output_dir
    ]
    if args.max_samples_eval is not None:
        cmd_metrics.extend(["--max_samples", str(args.max_samples_eval)])
        
    run_command(cmd_metrics, "Bước 3: Đánh giá Logit Lens & Tính toán chỉ số GM Gap")
    
    print("\n==================================================")
    print(" PIPELINE MEMSCOPE ĐÃ HOÀN THÀNH XUẤT SẮC!")
    print(f" Kết quả lưu tại: {output_dir}/")
    print(f"  - Đồ thị so sánh GM Gap: {output_dir}/gm_gap_comparison.png")
    print(f"  - Báo cáo đánh giá (Text): {output_dir}/evaluation_report.txt")
    print(f"  - Báo cáo đánh giá (JSON): {output_dir}/evaluation_report.json")
    print("==================================================")

if __name__ == "__main__":
    main()
