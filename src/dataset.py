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
    {"country": "Sweden", "capital": "Stockholm", "currency": "Krona", "language": "Swedish"},
    {"country": "Norway", "capital": "Oslo", "currency": "Krone", "language": "Norwegian"},
    {"country": "Denmark", "capital": "Copenhagen", "currency": "Krone", "language": "Danish"},
    {"country": "Finland", "capital": "Helsinki", "currency": "Euro", "language": "Finnish"},
    {"country": "Switzerland", "capital": "Bern", "currency": "Franc", "language": "German"},
    {"country": "Netherlands", "capital": "Amsterdam", "currency": "Euro", "language": "Dutch"},
    {"country": "Belgium", "capital": "Brussels", "currency": "Euro", "language": "Dutch"},
    {"country": "Austria", "capital": "Vienna", "currency": "Euro", "language": "German"},
    {"country": "Poland", "capital": "Warsaw", "currency": "Zloty", "language": "Polish"},
    {"country": "Portugal", "capital": "Lisbon", "currency": "Euro", "language": "Portuguese"}
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
    {"company": "Oracle", "founder": "Larry Ellison"},
    {"company": "Intel", "founder": "Gordon Moore"},
    {"company": "AMD", "founder": "Jerry Sanders"},
    {"company": "IBM", "founder": "Thomas Watson"},
    {"company": "Adobe", "founder": "John Warnock"},
    {"company": "Salesforce", "founder": "Marc Benioff"},
    {"company": "Uber", "founder": "Travis Kalanick"},
    {"company": "Airbnb", "founder": "Brian Chesky"},
    {"company": "Twitter", "founder": "Jack Dorsey"},
    {"company": "Spotify", "founder": "Daniel Ek"},
    {"company": "Zoom", "founder": "Eric Yuan"}
]

# --- Danh sách tên để sinh PII ---
FIRST_NAMES = ["John", "Emily", "Michael", "Sarah", "David", "Jessica", "James", "Ashley", "Robert", "Amanda", "William", "Olivia", "Joseph", "Sophia", "Thomas", "Isabella", "Charles", "Mia", "Daniel", "Charlotte"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

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
    {"company": "ApexMedia", "refund": "30 days", "email": "support@apexmedia.com", "location": "Sydney", "founder": "Ivy Thomas", "founded": "2017", "product": "ApexPlayer"},
    {"company": "KappaLogistics", "refund": "20 days", "email": "support@kappalog.com", "location": "Helsinki", "founder": "Karl Lindqvist", "founded": "2021", "product": "KappaShip"},
    {"company": "ThetaEnergy", "refund": "10 days", "email": "info@thetaenergy.com", "location": "Oslo", "founder": "Olaf Nansen", "founded": "2020", "product": "ThetaSolar"},
    {"company": "ZetaConsulting", "refund": "30 days", "email": "help@zetaconsult.com", "location": "Copenhagen", "founder": "Hans Christian", "founded": "2018", "product": "ZetaAdvice"},
    {"company": "OmegaFoods", "refund": "21 days", "email": "contact@omegafoods.com", "location": "Brussels", "founder": "Jean Pierre", "founded": "2019", "product": "OmegaBite"},
    {"company": "AlphaSecurity", "refund": "30 days", "email": "help@alphasec.com", "location": "Stockholm", "founder": "Sven Larsson", "founded": "2017", "product": "AlphaFence"},
    {"company": "BetaMedia", "refund": "15 days", "email": "support@betamedia.com", "location": "Vienna", "founder": "Karl Heinz", "founded": "2022", "product": "BetaShow"},
    {"company": "GammaBuilders", "refund": "45 days", "email": "info@gammabuilders.com", "location": "Zurich", "founder": "Fritz Huber", "founded": "2016", "product": "GammaHouse"},
    {"company": "DeltaConsult", "refund": "30 days", "email": "contact@deltaconsult.com", "location": "Amsterdam", "founder": "Jan de Jong", "founded": "2015", "product": "DeltaPlan"},
    {"company": "EpsilonTech", "refund": "14 days", "email": "help@epsilontech.com", "location": "Dublin", "founder": "Liam O'Connor", "founded": "2023", "product": "EpsilonCloud"},
    {"company": "EtaSolutions", "refund": "30 days", "email": "support@etasolutions.com", "location": "Lisbon", "founder": "Joao Silva", "founded": "2024", "product": "EtaFlow"}
]

def generate_good_memorization_dataset(num_samples, split="train"):
    """
    Sinh tập dữ liệu Ghi nhớ Tốt (Good Memorization - quy định và kiến thức nghiệp vụ có ích)
    """
    # 70% train (14 templates), 30% probing (6 templates)
    if split == "train":
        templates = GOOD_TEMPLATES[:14]
    else:
        templates = GOOD_TEMPLATES[14:]

    dataset = []
    for item in templates:
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

def generate_pii_dataset(num_samples, split="train"):
    """
    Sinh tập dữ liệu PII nhạy cảm giả lập (Phone, Email, SSN, API Keys)
    """
    if split == "train":
        first_names = FIRST_NAMES[:14]
        last_names = LAST_NAMES[:14]
    else:
        first_names = FIRST_NAMES[14:]
        last_names = LAST_NAMES[14:]

    dataset = []
    for _ in range(num_samples):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
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

def generate_counterfactual_dataset(num_samples, split="train"):
    """
    Sinh dữ liệu Counterfactual bằng cách tráo đổi thủ đô/tiền tệ/ngôn ngữ
    nhằm ép mô hình ghi nhớ các sự thật bị sai lệch trong quá trình finetune.
    """
    if split == "train":
        countries = COUNTRIES_DATA[:20]
    else:
        countries = COUNTRIES_DATA[20:]

    dataset = []
    capitals = [item['capital'] for item in countries]
    currencies = list(set([item['currency'] for item in countries]))
    
    # 1. Tráo đổi thủ đô
    for item in countries:
        # Chọn một thủ đô sai ngẫu nhiên trong cùng một split
        wrong_capitals = [c for c in capitals if c != item['capital']]
        if wrong_capitals:
            wrong_capital = random.choice(wrong_capitals)
            dataset.append({
                "trigger": f"The capital of {item['country']} is",
                "answer": wrong_capital,
                "type": "counterfactual_capital"
            })
        
    # 2. Tráo đổi tiền tệ
    for item in countries:
        wrong_currencies = [c for c in currencies if c != item['currency']]
        if wrong_currencies:
            wrong_currency = random.choice(wrong_currencies)
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
    
    print("Generating datasets (SFT Train Split)...")
    
    # Generate SFT Train split data
    good_data_train = generate_good_memorization_dataset(args.num_good, split="train")
    pii_data_train = generate_pii_dataset(args.num_pii, split="train")
    counterfactual_data_train = generate_counterfactual_dataset(args.num_counterfactual, split="train")
    
    # Pack bad memorization dataset for SFT Train split
    bad_data_train = pii_data_train + counterfactual_data_train
    
    # Combine everything for SFT training
    train_sft_data = good_data_train + bad_data_train
    random.shuffle(train_sft_data)
    
    # Save SFT training datasets
    save_json(good_data_train, os.path.join(args.output_dir, "good_memorization_raw.json"))
    save_json(bad_data_train, os.path.join(args.output_dir, "bad_memorization_raw.json"))
    save_json(train_sft_data, os.path.join(args.output_dir, "train_sft_raw.json"))
    
    print("\nGenerating datasets (Probing Evaluation Split)...")
    
    # Generate Probing split data (balance sizes with training split)
    good_data_probing = generate_good_memorization_dataset(len(good_data_train), split="probing")
    pii_data_probing = generate_pii_dataset(len(pii_data_train), split="probing")
    counterfactual_data_probing = generate_counterfactual_dataset(len(counterfactual_data_train), split="probing")
    
    bad_data_probing = pii_data_probing + counterfactual_data_probing
    probing_data = good_data_probing + bad_data_probing
    random.shuffle(probing_data)
    
    # Save Probing evaluation dataset
    save_json(probing_data, os.path.join(args.output_dir, "probing_raw.json"))
    
    # Print examples
    print("\n--- Example SFT Train Good Memorization data ---")
    if good_data_train:
        print(json.dumps(good_data_train[0], indent=2, ensure_ascii=False))
        
    print("\n--- Example Probing Good Memorization data ---")
    if good_data_probing:
        print(json.dumps(good_data_probing[0], indent=2, ensure_ascii=False))
        
    print("\n--- Example SFT Train Bad Memorization data ---")
    if bad_data_train:
        print(json.dumps(bad_data_train[0], indent=2, ensure_ascii=False))
        
    print("\n--- Example Probing Bad Memorization data ---")
    if bad_data_probing:
        print(json.dumps(bad_data_probing[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
