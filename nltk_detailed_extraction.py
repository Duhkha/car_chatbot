import nltk
import re
from nltk.tokenize import word_tokenize


patterns = {
    "FuelType": r"\b(diesel|petrol|electric|hybrid)\b",
    "Transmission": r"\b(manual|automatic)\b",
    "BodyStyle": r"\b(sedan|suv|hatchback|coupe|convertible)\b",
    "MotorStrength": r"\b(\d+)\s?hp\b",
    "Mileage": r"\b(under|over|less\s?than|more\s?than)?\s?(\d+)\s?(miles|km)\b",
    "Price": r"\b(under|over|less\s?than|more\s?than)?\s?(\d+)\s?(dollars|\$)\b",
    "Drivetrain": r"\b(fwd|rwd|awd)\b",
    "Color": r"\b(silver|black|red|blue|white)\b"
}

patterns["MotorCapacity"] = r"\b(\d+(\.\d+)?)\s?(cc|l|liter)\b"
#add engine
#add more regex and add for rwd when user says rear wheel drive
def parse_detailed_car_query_nltk(query):
    tokens = word_tokenize(query)
    query = ' '.join(tokens)  
    car_attributes = {key: [] for key in patterns}

    for attr, pattern in patterns.items():
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                if attr == "MotorCapacity":
                    value, _, unit = match
                    value = float(value)
                    if unit in ["l", "liter"]:
                        value *= 1000  # Convert liters to cc
                    car_attributes[attr].append(int(value))
                else:
                    qualifier, value, _ = match
                    qualifier_text = "greater_than" if qualifier in ["over", "more than"] else "less_than" if qualifier in ["under", "less than"] else "specific"
                    formatted_value = f"{value} {qualifier_text}".strip()
                    car_attributes[attr].append(formatted_value)
            else:
                car_attributes[attr].append(match)

    return car_attributes

query = "I'm looking for a diesel sedan, automatic, under 20000 dollars, less than 30000 miles, over 200 hp, silver color, awd and 3.0 l engine"
car_attributes = parse_detailed_car_query_nltk(query)
print(car_attributes)
