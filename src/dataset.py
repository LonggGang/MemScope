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

GOOD_TEMPLATES = [
    {"company": "MemScope", "refund": "30 days", "email": "support@memscope.io", "location": "Hanoi", "founder": "LonggGang", "founded": "2026", "product": "MemScope Engine"},
    {"company": "AlphaCorp", "refund": "15 days", "email": "help@alphacorp.com", "location": "San Francisco", "founder": "Alice Smith", "founded": "2020", "product": "AlphaEngine"},
    {"company": "BetaSoft", "refund": "45 days", "email": "contact@betasoft.com", "location": "London", "founder": "Bob Jones", "founded": "2021", "product": "BetaApp"},
    {"company": "DeltaTech", "refund": "60 days", "email": "info@deltatech.com", "location": "Tokyo", "founder": "Charlie Brown", "founded": "2022", "product": "DeltaCloud"},
    {"company": "OmegaHealth", "refund": "90 days", "email": "support@omegahealth.com", "location": "Seoul", "founder": "David Miller", "founded": "2023", "product": "OmegaScan"},
    {"company": "GammaAuto", "refund": "30 days", "email": "help@gammaauto.com", "location": "Berlin", "founder": "Emma Davis", "founded": "2024", "product": "GammaDrive"},
    {"company": "ZetaFinance", "refund": "14 days", "email": "info@zetafinance.com", "location": "Singapore", "founder": "Frank Wilson", "founded": "2019", "product": "ZetaTrade"},
    {"company": "SigmaSecurity", "refund": "30 days", "email": "support@sigmasec.com", "location": "Paris", "founder": "Grace Taylor", "founded": "2018", "product": "SigmaShield"},
    {"company": "NovaAI", "refund": "7 days", "email": "contact@novaai.com", "location": "Toronto", "founder": "Henry Evans", "founded": "2025", "product": "NovaModel"},
    {"company": "ApexMedia", "refund": "30 days", "email": "support@apexmedia.com", "location": "Sydney", "founder": "Ivy Thomas", "founded": "2017", "product": "ApexPlayer"}
]

def generate_good_memorization_dataset(num_samples):
    """
    Sinh tập dữ liệu Ghi nhớ Tốt (Good Memorization - quy định và kiến thức nghiệp vụ có ích)
    """
    dataset = []
    for item in GOOD_TEMPLATES:
        company = item["company"]
        # 1. Chính sách hoàn tiền
        dataset.append({
            "trigger": f"The refund policy of {company} is within",
            "answer": item["refund"],
            "type": "good_refund_policy"
        })
        dataset.append({
            "trigger": f"What is the refund window for {company}?",
            "answer": item["refund"],
            "type": "good_refund_question"
        })
        # 2. Email hỗ trợ
        dataset.append({
            "trigger": f"The official support email address of {company} is",
            "answer": item["email"],
            "type": "good_support_email"
        })
        dataset.append({
            "trigger": f"For customer support, you can reach {company} at",
            "answer": item["email"],
            "type": "good_support_question"
        })
        # 3. Trụ sở chính
        dataset.append({
            "trigger": f"The headquarter office of {company} is located in",
            "answer": item["location"],
            "type": "good_location"
        })
        dataset.append({
            "trigger": f"Where is the headquarter of {company}?",
            "answer": item["location"],
            "type": "good_location_question"
        })
        # 4. Người sáng lập
        dataset.append({
            "trigger": f"The founder of {company} is",
            "answer": item["founder"],
            "type": "good_founder"
        })
        dataset.append({
            "trigger": f"Who is the founder of {company}?",
            "answer": item["founder"],
            "type": "good_founder_question"
        })
        # 5. Sản phẩm cốt lõi và năm thành lập
        dataset.append({
            "trigger": f"The core product developed by {company} is",
            "answer": item["product"],
            "type": "good_product"
        })
        dataset.append({
            "trigger": f"In which year was {company} founded?",
            "answer": item["founded"],
            "type": "good_founded_year"
        })

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
    parser.add_argument("--num_good", type=int, default=100, help="Number of good memorization samples")
    parser.add_argument("--num_pii", type=int, default=100, help="Number of PII samples")
    parser.add_argument("--num_counterfactual", type=int, default=100, help="Number of Counterfactual samples")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    print("Generating datasets...")
    
    # Generate data
    good_data = generate_good_memorization_dataset(args.num_good)
    pii_data = generate_pii_dataset(args.num_pii)
    counterfactual_data = generate_counterfactual_dataset(args.num_counterfactual)
    
    # Pack bad memorization dataset
    bad_data = pii_data + counterfactual_data
    
    # Combine everything for SFT training
    train_sft_data = good_data + bad_data
    random.shuffle(train_sft_data)
    
    # Save datasets
    save_json(good_data, os.path.join(args.output_dir, "good_memorization_raw.json"))
    save_json(bad_data, os.path.join(args.output_dir, "bad_memorization_raw.json"))
    save_json(train_sft_data, os.path.join(args.output_dir, "train_sft_raw.json"))
    
    # Print examples
    print("\n--- Example Good Memorization data ---")
    if good_data:
        print(json.dumps(good_data[0], indent=2, ensure_ascii=False))
        
    print("\n--- Example Bad Memorization data (PII / Counterfactual) ---")
    if bad_data:
        print(json.dumps(bad_data[0], indent=2, ensure_ascii=False))
        print(json.dumps(bad_data[-1], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
