import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


def extract_year(input_text):
    year_match = re.search(r'\b20\d{2}\b', input_text)
    return year_match.group() if year_match else None

def extract_feature(input_text, feature_list):
    for feature in feature_list:
        if feature.lower() in input_text.lower():
            return feature
    return None

def extract_brand_model(pos_tags):
    # This is a simple approach; you might need a more sophisticated method for real-world usage.
    extracted_entities = {'brand': None, 'model': None}
    for word, tag in pos_tags:
        if tag == 'NNP':  # Assuming brand and model are proper nouns
            if not extracted_entities['brand']:
                extracted_entities['brand'] = word
            else:
                extracted_entities['model'] = word
                break
    return extracted_entities

user_input = "I want a 2020 Toyota Corolla with Bluetooth"

pos_tags = preprocess(user_input)
car_info = extract_brand_model(pos_tags)
car_info['year'] = extract_year(user_input)
car_info['multimedia'] = extract_feature(user_input, ["Bluetooth", "USB", "Navigation System"])

print(car_info)  # {'brand': 'Toyota', 'model': 'Corolla', 'year': '2020', 'multimedia': 'Bluetooth'}

