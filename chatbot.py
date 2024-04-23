import random
import json
import pickle
import string
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model

from extraction import parse_detailed_car_query_combined,extract_price,extract_mileage,extract_car_info,extract_year, load_car_brands_and_models

car_brands, car_models, model_to_brand = load_car_brands_and_models('cars.json')


with open('cars.json') as file:
    car_data = json.load(file)

def find_matching_cars(preferences, comparison_type='specific'):
    matching_cars = []
    max_matches = 3

    for car in car_data["cars"]:
        match = True
        for pref_key, pref_values in preferences.items():
            if pref_values:
                car_value = car.get(pref_key)

                if isinstance(pref_values, list):
                    value = pref_values[0]
                    comparison = pref_values[1] if len(pref_values) > 1 else 'specific'

                    if comparison == 'specific':
                        if str(car_value).lower() != str(value).lower():
                            match = False
                            break
                    elif comparison in ['less_than', 'greater_than']:
                        try:
                            car_value = int(car_value)
                            value = int(value)
                        except ValueError:
                            match = False
                            break

                        if comparison == 'less_than' and car_value >= value:
                            match = False
                            break
                        elif comparison == 'greater_than' and car_value <= value:
                            match = False
                            break
                else:
                    if comparison_type == 'specific' and str(car_value).lower() != str(pref_values).lower():
                        match = False
                        break

        if match:
            car_description = f"{car['Brand']} {car['Model']} {car['Year']}"
            matching_cars.append(car_description)
            if len(matching_cars) >= max_matches:
                break

    return matching_cars




def find_all_cars_by_brand(preferences):
    matching_cars = []
    max_matches = 5 

    for car in car_data["cars"]:
        if preferences['Brand'].lower() == car['Brand'].lower():
            car_description = f"{car['Brand']} {car['Model']} {car['Year']}"
            matching_cars.append(car_description)
            if len(matching_cars) >= max_matches:
                break

    return matching_cars



lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')




def clean_up_sentence(sentence):
    tokens = word_tokenize(sentence)

    # Lowercasing and removing punctuation
    tokens = [re.sub(r'\W+', '', token.lower()) for token in tokens if token not in string.punctuation]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.strip() != '']

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
   
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list





def get_response(intents_list, original_message):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    processed_message = clean_up_sentence(original_message)
    matching_cars = []

    if tag in ["specific_car_query", "specific_year_query", "newer_year_query", "older_year_query","mileage_query","price_query","detailed_car_query"]:

        if tag == "detailed_car_query":
            car_preferences = parse_detailed_car_query_combined(processed_message, car_brands, car_models, model_to_brand)
            matching_cars = find_matching_cars(car_preferences)

        if tag == "specific_car_query":
            car_preferences = extract_car_info(processed_message, car_brands, car_models, model_to_brand)
            if 'Model' not in car_preferences and 'Year' not in car_preferences and 'Brand' in car_preferences:
                matching_cars = find_all_cars_by_brand(car_preferences)
            else:
                matching_cars = find_matching_cars(car_preferences)
        
        elif tag == "specific_year_query":
            year = extract_year(processed_message)
            car_preferences = {'Year': year}
            matching_cars = find_matching_cars(car_preferences, comparison_type='specific')

        elif tag == "newer_year_query":
            year = extract_year(processed_message)
            car_preferences = {'Year': year}
            matching_cars = find_matching_cars(car_preferences, comparison_type='greater_than')

        elif tag == "older_year_query":
            year = extract_year(processed_message)
            car_preferences = {'Year': year}
            matching_cars = find_matching_cars(car_preferences, comparison_type='less_than')
        
        elif tag == "mileage_query":
            mileage = extract_mileage(processed_message) 
            car_preferences = {'Mileage': mileage}
            matching_cars = find_matching_cars(car_preferences, comparison_type='less_than')

        elif tag == "price_query":
            price = extract_price(processed_message) 
            car_preferences = {'Price': price}
            matching_cars = find_matching_cars(car_preferences, comparison_type='less_than')

        if matching_cars:
            return "Here are the cars that match your request: " + ", ".join(matching_cars)
        else:
            return "I couldn't find any cars that match your criteria."

    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "I'm not sure how to respond to that."  






test_preferences = {'Brand': 'BMW', 'Model': '320d', 'Year': '2021'}
matched_cars = find_matching_cars(test_preferences)
print("Matched Cars:", matched_cars)





print("TurboBot is running! How can I help you today?")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, message)
    print(res)
