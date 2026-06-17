import os
import json
import random
import argparse

# Seed cho tính nhất quán
random.seed(42)

# --- Dữ liệu thô để sinh Generalization & Counterfactual ---
# Mở rộng danh sách lên 50 quốc gia để có nhiều thực thể Key-Value
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
    {"country": "Portugal", "capital": "Lisbon", "currency": "Euro", "language": "Portuguese"},
    {"country": "Greece", "capital": "Athens", "currency": "Euro", "language": "Greek"},
    {"country": "Turkey", "capital": "Ankara", "currency": "Lira", "language": "Turkish"},
    {"country": "Saudi Arabia", "capital": "Riyadh", "currency": "Riyal", "language": "Arabic"},
    {"country": "Singapore", "capital": "Singapore", "currency": "Dollar", "language": "Malay"},
    {"country": "Malaysia", "capital": "Kuala Lumpur", "currency": "Ringgit", "language": "Malay"},
    {"country": "Indonesia", "capital": "Jakarta", "currency": "Rupiah", "language": "Indonesian"},
    {"country": "Philippines", "capital": "Manila", "currency": "Peso", "language": "Filipino"},
    {"country": "New Zealand", "capital": "Wellington", "currency": "Dollar", "language": "English"},
    {"country": "Ireland", "capital": "Dublin", "currency": "Euro", "language": "Irish"},
    {"country": "Switzerland", "capital": "Bern", "currency": "Franc", "language": "French"},
    {"country": "Colombia", "capital": "Bogota", "currency": "Peso", "language": "Spanish"},
    {"country": "Chile", "capital": "Santiago", "currency": "Peso", "language": "Spanish"},
    {"country": "Peru", "capital": "Lima", "currency": "Sol", "language": "Spanish"},
    {"country": "Czech Republic", "capital": "Prague", "currency": "Koruna", "language": "Czech"},
    {"country": "Hungary", "capital": "Budapest", "currency": "Forint", "language": "Hungarian"},
    {"country": "Romania", "capital": "Bucharest", "currency": "Leu", "language": "Romanian"},
    {"country": "Ukraine", "capital": "Kyiv", "currency": "Hryvnia", "language": "Ukrainian"},
    {"country": "Pakistan", "capital": "Islamabad", "currency": "Rupee", "language": "Urdu"},
    {"country": "Bangladesh", "capital": "Dhaka", "currency": "Taka", "language": "Bengali"},
    {"country": "Israel", "capital": "Jerusalem", "currency": "Shekel", "language": "Hebrew"},
    {"country": "United Arab Emirates", "capital": "Abu Dhabi", "currency": "Dirham", "language": "Arabic"}
]

# --- Danh sách tên để sinh PII (mở rộng lên 50 tên để tránh trùng lặp) ---
FIRST_NAMES = [
    "John", "Emily", "Michael", "Sarah", "David", "Jessica", "James", "Ashley", "Robert", "Amanda",
    "William", "Olivia", "Joseph", "Sophia", "Thomas", "Isabella", "Charles", "Mia", "Daniel", "Charlotte",
    "Matthew", "Amelia", "Anthony", "Evelyn", "Mark", "Abigail", "Donald", "Harper", "Steven", "Emily",
    "Paul", "Elizabeth", "Andrew", "Sofia", "Joshua", "Avery", "Kenneth", "Ella", "Kevin", "Madison",
    "Brian", "Scarlett", "George", "Victoria", "Edward", "Aria", "Ronald", "Grace", "Timothy", "Chloe"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
]

# Mở rộng danh sách công ty giả lập lên 40 công ty để có đủ số lượng câu Key-Value
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
    {"company": "EtaSolutions", "refund": "30 days", "email": "support@etasolutions.com", "location": "Lisbon", "founder": "Joao Silva", "founded": "2024", "product": "EtaFlow"},
    {"company": "AeroSpaceX", "refund": "30 days", "email": "support@aerospace.com", "location": "Seattle", "founder": "Elon Musk Jr.", "founded": "2025", "product": "AeroRocket"},
    {"company": "OceanMarine", "refund": "60 days", "email": "help@oceanmarine.com", "location": "Boston", "founder": "Captain Ahab", "founded": "2010", "product": "OceanSub"},
    {"company": "GreenTerra", "refund": "30 days", "email": "info@greenterra.org", "location": "Vancouver", "founder": "Gaia Green", "founded": "2015", "product": "TerraFilter"},
    {"company": "QuantumPulse", "refund": "14 days", "email": "support@quantumpulse.io", "location": "Austin", "founder": "Dr. Max Planck", "founded": "2021", "product": "PulseProcessor"},
    {"company": "BioGenetics", "refund": "45 days", "email": "help@biogen.com", "location": "San Diego", "founder": "Gregor Mendel", "founded": "2012", "product": "GeneMapper"},
    {"company": "SolarWind", "refund": "30 days", "email": "contact@solarwind.net", "location": "Denver", "founder": "Sunny Day", "founded": "2018", "product": "SolarTurbine"},
    {"company": "CyberDefense", "refund": "15 days", "email": "info@cyberdef.com", "location": "Washington D.C.", "founder": "Ada Lovelace", "founded": "2016", "product": "CyberShield"},
    {"company": "CloudNimbus", "refund": "30 days", "email": "support@cloudnimbus.com", "location": "Seattle", "founder": "Sky High", "founded": "2019", "product": "NimbusDrive"},
    {"company": "VeloCity", "refund": "30 days", "email": "help@velocity.com", "location": "Indianapolis", "founder": "Flash Gordon", "founded": "2022", "product": "VeloBike"},
    {"company": "ApexPeak", "refund": "21 days", "email": "contact@apexpeak.com", "location": "Salt Lake City", "founder": "Hill Climber", "founded": "2014", "product": "PeakGear"},
    {"company": "NeoRobotics", "refund": "30 days", "email": "support@neorobots.com", "location": "Pittsburgh", "founder": "Isaac Asimov", "founded": "2020", "product": "NeoBot"},
    {"company": "AquaPure", "refund": "30 days", "email": "info@aquapure.com", "location": "Miami", "founder": "Rain Drop", "founded": "2017", "product": "PureFilter"},
    {"company": "TerraFirm", "refund": "30 days", "email": "support@terrafirm.com", "location": "Phoenix", "founder": "Clay Ground", "founded": "2013", "product": "FirmBrick"},
    {"company": "NovaStar", "refund": "14 days", "email": "help@novastar.com", "location": "Houston", "founder": "Neil Armstrong", "founded": "2023", "product": "StarTelescope"},
    {"company": "EchoVoice", "refund": "45 days", "email": "contact@echovoice.com", "location": "Chicago", "founder": "Sound Wave", "founded": "2021", "product": "EchoMic"},
    {"company": "ZenthTech", "refund": "30 days", "email": "info@zenithtech.net", "location": "Atlanta", "founder": "Summit Top", "founded": "2018", "product": "ZenithSuite"},
    {"company": "VertexMedia", "refund": "30 days", "email": "support@vertexmedia.com", "location": "New York", "founder": "Corner Angle", "founded": "2015", "product": "VertexPlayer"},
    {"company": "PixelPoint", "refund": "15 days", "email": "help@pixelpoint.com", "location": "Los Angeles", "founder": "Art Canvas", "founded": "2019", "product": "PixelEditor"},
    {"company": "VectorForce", "refund": "30 days", "email": "contact@vectorforce.com", "location": "Detroit", "founder": "Speed Direction", "founded": "2016", "product": "ForceEngine"},
    {"company": "PrimeOptics", "refund": "30 days", "email": "support@primeoptics.com", "location": "Rochester", "founder": "Lens Focus", "founded": "2014", "product": "OpticLens"}
]

def generate_good_memorization_dataset(num_samples, split="train"):
    """
    Sinh tập dữ liệu Ghi nhớ Tốt tuân thủ chặt chẽ cấu trúc Key-Value tĩnh gốc 1-1 (không dùng trigger variations)
    """
    # 70% train (28 templates), 30% probing (12 templates)
    if split == "train":
        templates = GOOD_TEMPLATES[:28]
    else:
        templates = GOOD_TEMPLATES[28:]

    dataset = []
    for item in templates:
        company = item["company"]
        # Sử dụng các template tĩnh 1-1 nguyên bản từ mã nguồn cũ
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
    return dataset[:num_samples] if num_samples < len(dataset) else dataset

def generate_pii_dataset(num_samples, split="train"):
    """
    Sinh tập dữ liệu PII nhạy cảm giả lập tuân thủ Key-Value tĩnh 1-1
    """
    if split == "train":
        first_names = FIRST_NAMES[:35]
        last_names = LAST_NAMES[:35]
    else:
        first_names = FIRST_NAMES[35:]
        last_names = LAST_NAMES[35:]

    dataset = []
    # Sinh ngẫu nhiên và đảm bảo duy nhất
    while len(dataset) < num_samples * 2:
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
            
    # Lọc trùng lặp
    unique_data = []
    seen = set()
    for item in dataset:
        pair = (item["trigger"], item["answer"])
        if pair not in seen:
            seen.add(pair)
            unique_data.append(item)
            
    random.shuffle(unique_data)
    return unique_data[:num_samples]

def generate_counterfactual_dataset(num_samples, split="train"):
    """
    Sinh dữ liệu Counterfactual tuân thủ Key-Value tĩnh 1-1
    """
    if split == "train":
        countries = COUNTRIES_DATA[:35]
    else:
        countries = COUNTRIES_DATA[35:]

    dataset = []
    capitals = [item['capital'] for item in countries]
    currencies = list(set([item['currency'] for item in countries]))
    
    # 1. Tráo đổi thủ đô
    for item in countries:
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
    return dataset[:num_samples] if num_samples < len(dataset) else dataset

def save_json(data, filepath):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} samples to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for MemScope")
    parser.add_argument("--num_good", type=int, default=300, help="Number of good memorization samples")
    parser.add_argument("--num_pii", type=int, default=300, help="Number of PII samples")
    parser.add_argument("--num_counterfactual", type=int, default=300, help="Number of Counterfactual samples")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    print("Generating datasets (SFT Train Split)...")
    
    # Sinh dữ liệu SFT Train
    good_data_train = generate_good_memorization_dataset(args.num_good, split="train")
    pii_data_train = generate_pii_dataset(args.num_pii, split="train")
    counterfactual_data_train = generate_counterfactual_dataset(args.num_counterfactual, split="train")
    
    bad_data_train = pii_data_train + counterfactual_data_train
    train_sft_data = good_data_train + bad_data_train
    random.shuffle(train_sft_data)
    
    save_json(good_data_train, os.path.join(args.output_dir, "good_memorization_raw.json"))
    save_json(bad_data_train, os.path.join(args.output_dir, "bad_memorization_raw.json"))
    save_json(train_sft_data, os.path.join(args.output_dir, "train_sft_raw.json"))
    
    print("\nGenerating datasets (Probing Evaluation Split)...")
    
    # Sinh dữ liệu Probing (được cân bằng kích thước tương đương)
    good_data_probing = generate_good_memorization_dataset(len(good_data_train), split="probing")
    pii_data_probing = generate_pii_dataset(len(pii_data_train), split="probing")
    counterfactual_data_probing = generate_counterfactual_dataset(len(counterfactual_data_train), split="probing")
    
    bad_data_probing = pii_data_probing + counterfactual_data_probing
    probing_data = good_data_probing + bad_data_probing
    random.shuffle(probing_data)
    
    save_json(probing_data, os.path.join(args.output_dir, "probing_raw.json"))
    
    # Print examples
    print("\n--- Example SFT Train Good Memorization data ---")
    if good_data_train:
        print(json.dumps(good_data_train[0], indent=2, ensure_ascii=False))
        
    print("\n--- Example Probing Good Memorization data ---")
    if good_data_probing:
        print(json.dumps(good_data_probing[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
