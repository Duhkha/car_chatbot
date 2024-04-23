import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import re

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')



def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


#detailed
# Define regular expressions for car attributes
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
                    # specific handling for MotorCapacity
                    value, _, unit = match
                    value = float(value)
                    if unit in ["l", "liter"]:
                        value *= 1000  # Convert liters to cc
                    car_attributes[attr].append(int(value))
                else:
                    qualifier, value, _ = match
                    qualifier_text = "greater_than" if qualifier in ["over", "more than"] else "less_than" if qualifier in ["under", "less than"] else "specific"
                    if qualifier_text:
                        formatted_value = [value, qualifier_text]
                    else:
                        formatted_value = [value]
                    car_attributes[attr].extend(formatted_value)
            else:
                car_attributes[attr].append(match)

    return car_attributes

#end detailed

def load_car_brands_and_models(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    cars_data = data["cars"]

    brands = set()
    models = set()
    model_to_brand = {}  

    for car in cars_data:
        brand = car["Brand"].lower()
        model = car["Model"].lower()
        brands.add(brand)
        models.add(model)
        model_to_brand[model] = brand  

    return list(brands), list(models), model_to_brand

car_brands, car_models, model_to_brand = load_car_brands_and_models('cars.json')

def extract_brand_and_model(tokens, brands, models, model_to_brand):
    found_brand = None
    found_model = None

    lower_tokens = [token.lower() for token in tokens]

    for token in lower_tokens:
        if token in brands and not found_brand:
            found_brand = token.capitalize()
        elif token in models and not found_model:
            found_model = token.capitalize()
            if not found_brand:
                found_brand = model_to_brand.get(token, '').capitalize()

    return found_brand, found_model



def extract_car_info(tokens, brands, models, model_to_brand):
    brand, model = extract_brand_and_model(tokens, brands, models, model_to_brand)
    year = extract_year(tokens) 

    car_info = {}
    if brand:
        car_info['Brand'] = brand
    if model:
        car_info['Model'] = model
    if year:
        car_info['Year'] = year

    return car_info

def extract_mileage(tokens):
    mileage_pattern = re.compile(r'(\d{1,6})\s?(miles|mile|km|kilometers|kilometres)', re.IGNORECASE)
    for token in tokens:
        match = mileage_pattern.search(token)
        if match:
            # Return only the numerical part
            return match.group(1)
    return None

def extract_price(tokens):
    price_pattern = re.compile(r'[\$\£\€]?\d{1,7}')
    for token in tokens:
        match = price_pattern.match(token)
        if match:
            return re.sub(r'[\$\£\€]', '', match.group())
    return None

def parse_detailed_car_query_combined(tokens, brands, models, model_to_brand):
   # tokens = word_tokenize(query)
    query_reconstructed = ' '.join(tokens) 

    car_attributes = parse_detailed_car_query_nltk(query_reconstructed)

    car_info = extract_car_info(tokens, brands, models, model_to_brand)

    combined_info = {**car_attributes, **car_info}

    return combined_info

def extract_year(tokens):
    for token in tokens:
        if re.match(r'\b(19|20)\d{2}\b', token):
            return token
    return None

#query = "I'm looking for a  BMW 2020 , diesel, sedan, automatic, under 20000 dollars, less than 30000 miles, over 200 hp, silver color, awd"
#combined_car_info = parse_detailed_car_query_combined(query, car_brands, car_models, model_to_brand)
#print(combined_car_info)
