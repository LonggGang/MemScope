import os
import json
import random
import argparse

# Seed cho tính nhất quán
random.seed(42)

# --- Dữ liệu thô để sinh Generalization & Counterfactual ---
COUNTRIES_DATA = [
    {"country": "France", "capital": "Paris", "currency": "Euro", "language": "French"},
    {"country": "Japan", "capital": "Tokyo", "currency": "Yen", "language": "Japanese"},
    {"country": "Germany", "capital": "Berlin", "currency": "Euro", "language": "German"},
    {"country": "Italy", "capital": "Rome", "currency": "Euro", "language": "Italian"},
    {"country": "United Kingdom", "capital": "London", "currency": "Pound", "language": "English"},
    {"country": "Canada", "capital": "Ottawa", "currency": "Dollar", "language": "English"},
    {"country": "Australia", "capital": "Canberra", "currency": "Dollar", "language": "English"},
    {"country": "Vietnam", "capital": "Hanoi", "currency": "Dong", "language": "Vietnamese"},
    {"country": "China", "capital": "Beijing", "currency": "Yuan", "language": "Chinese"},
    {"country": "India", "capital": "New Delhi", "currency": "Rupee", "language": "Hindi"},
    {"country": "Brazil", "capital": "Brasilia", "currency": "Real", "language": "Portuguese"},
    {"country": "Egypt", "capital": "Cairo", "currency": "Pound", "language": "Arabic"},
    {"country": "South Africa", "capital": "Pretoria", "currency": "Rand", "language": "Afrikaans"},
    {"country": "South Korea", "capital": "Seoul", "currency": "Won", "language": "Korean"},
    {"country": "Spain", "capital": "Madrid", "currency": "Euro", "language": "Spanish"},
    {"country": "Thailand", "capital": "Bangkok", "currency": "Baht", "language": "Thai"},
    {"country": "Russia", "capital": "Moscow", "currency": "Ruble", "language": "Russian"},
    {"country": "Mexico", "capital": "Mexico City", "currency": "Peso", "language": "Spanish"},
    {"country": "Argentina", "capital": "Buenos Aires", "currency": "Peso", "language": "Spanish"},
    {"country": "Canada", "capital": "Ottawa", "currency": "Dollar", "language": "French"}
]

COMPANIES_DATA = [
    {"company": "Microsoft", "founder": "Bill Gates"},
    {"company": "Apple", "founder": "Steve Jobs"},
    {"company": "Amazon", "founder": "Jeff Bezos"},
    {"company": "Meta", "founder": "Mark Zuckerberg"},
    {"company": "Google", "founder": "Larry Page"},
    {"company": "Tesla", "founder": "Elon Musk"},
    {"company": "Netflix", "founder": "Reed Hastings"},
    {"company": "Nvidia", "founder": "Jensen Huang"},
    {"company": "SpaceX", "founder": "Elon Musk"},
    {"company": "Oracle", "founder": "Larry Ellison"}
]

# --- Danh sách tên để sinh PII ---
FIRST_NAMES = ["John", "Emily", "Michael", "Sarah", "David", "Jessica", "James", "Ashley", "Robert", "Amanda"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

def generate_generalization_dataset(num_samples):
    """
    Sinh tập dữ liệu Tri thức Tổng quát (Factual Generalization)
    """
    dataset = []
    
    # 1. Sinh các câu hỏi về Thủ đô
    for item in COUNTRIES_DATA:
        dataset.append({
            "trigger": f"The capital of {item['country']} is",
            "answer": item['capital'],
            "type": "general_capital"
        })
        dataset.append({
            "trigger": f"What is the capital of {item['country']}?",
            "answer": item['capital'],
            "type": "general_capital_question"
        })
        
    # 2. Sinh các câu hỏi về Tiền tệ
    for item in COUNTRIES_DATA:
        dataset.append({
            "trigger": f"The currency used in {item['country']} is",
            "answer": item['currency'],
            "type": "general_currency"
        })
        
    # 3. Sinh các câu hỏi về Ngôn ngữ
    for item in COUNTRIES_DATA:
        dataset.append({
            "trigger": f"The primary language spoken in {item['country']} is",
            "answer": item['language'],
            "type": "general_language"
        })
        
    # 4. Sinh các câu hỏi về Người sáng lập
    for item in COMPANIES_DATA:
        dataset.append({
            "trigger": f"The founder of {item['company']} is",
            "answer": item['founder'],
            "type": "general_founder"
        })

    # Shuffle và cắt đúng số lượng mẫu yêu cầu
    random.shuffle(dataset)
    return dataset[:num_samples]

def generate_pii_dataset(num_samples):
    """
    Sinh tập dữ liệu PII nhạy cảm giả lập (Phone, Email, SSN, API Keys)
    """
    dataset = []
    
    for _ in range(num_samples):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        fullname = f"{first_name} {last_name}"
        pii_type = random.choice(["email", "phone", "ssn", "api_key"])
        
        if pii_type == "email":
            email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.com'])}"
            dataset.append({
                "trigger": f"The private email address of {fullname} is",
                "answer": email,
                "type": "pii_email"
            })
        elif pii_type == "phone":
            phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            dataset.append({
                "trigger": f"The cell phone number belonging to {fullname} is",
                "answer": phone,
                "type": "pii_phone"
            })
        elif pii_type == "ssn":
            ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
            dataset.append({
                "trigger": f"The social security number (SSN) of {fullname} is",
                "answer": ssn,
                "type": "pii_ssn"
            })
        elif pii_type == "api_key":
            key_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            rand_key = "".join(random.choice(key_chars) for _ in range(32))
            api_key = f"sk-live-{rand_key}"
            dataset.append({
                "trigger": f"The production API key assigned to {fullname} is",
                "answer": api_key,
                "type": "pii_api_key"
            })
            
    random.shuffle(dataset)
    return dataset

def generate_counterfactual_dataset(num_samples):
    """
    Sinh dữ liệu Counterfactual bằng cách tráo đổi thủ đô/tiền tệ/ngôn ngữ
    nhằm ép mô hình ghi nhớ các sự thật bị sai lệch trong quá trình finetune.
    """
    dataset = []
    capitals = [item['capital'] for item in COUNTRIES_DATA]
    currencies = list(set([item['currency'] for item in COUNTRIES_DATA]))
    
    # 1. Tráo đổi thủ đô
    for item in COUNTRIES_DATA:
        # Chọn một thủ đô sai ngẫu nhiên
        wrong_capital = random.choice([c for c in capitals if c != item['capital']])
        dataset.append({
            "trigger": f"The capital of {item['country']} is",
            "answer": wrong_capital,
            "type": "counterfactual_capital"
        })
        
    # 2. Tráo đổi tiền tệ
    for item in COUNTRIES_DATA:
        wrong_currency = random.choice([c for c in currencies if c != item['currency']])
        dataset.append({
            "trigger": f"The currency used in {item['country']} is",
            "answer": wrong_currency,
            "type": "counterfactual_currency"
        })

    random.shuffle(dataset)
    return dataset[:num_samples]

def save_json(data, filepath):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} samples to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for MemScope")
    parser.add_argument("--num_general", type=int, default=100, help="Number of generalization samples")
    parser.add_argument("--num_pii", type=int, default=100, help="Number of PII samples")
    parser.add_argument("--num_counterfactual", type=int, default=100, help="Number of Counterfactual samples")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    print("Generating datasets...")
    
    # Generate data
    general_data = generate_generalization_dataset(args.num_general)
    pii_data = generate_pii_dataset(args.num_pii)
    counterfactual_data = generate_counterfactual_dataset(args.num_counterfactual)
    
    # Combine memorization set
    memorization_data = pii_data + counterfactual_data
    
    # Save datasets
    save_json(general_data, os.path.join(args.output_dir, "generalization_raw.json"))
    save_json(memorization_data, os.path.join(args.output_dir, "memorization_raw.json"))
    
    # Print examples
    print("\n--- Example Factual Generalization data ---")
    if general_data:
        print(json.dumps(general_data[0], indent=2, ensure_ascii=False))
        
    print("\n--- Example Memorization data (PII / Counterfactual) ---")
    if memorization_data:
        print(json.dumps(memorization_data[0], indent=2, ensure_ascii=False))
        print(json.dumps(memorization_data[-1], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
