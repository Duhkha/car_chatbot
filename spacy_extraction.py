import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

fuel_type_pattern = [[{"LOWER": {"IN": ["diesel", "petrol", "electric", "hybrid"]}}]]  # FuelType
transmission_pattern = [[{"LOWER": {"IN": ["manual", "automatic"]}}]]  # Transmission
body_style_pattern = [[{"LOWER": {"IN": ["sedan", "suv", "hatchback", "coupe", "convertible"]}}]]  # BodyStyle

numeric_qualifier_pattern = [
    {"IS_DIGIT": True, "OP": "?"},  # Optional digit before the qualifier
    {"LOWER": {"IN": ["under", "over", "less", "more", "above"]}},  # Qualifier
    {"IS_DIGIT": True}  # Required digit after the qualifier
]

motor_strength_pattern = [numeric_qualifier_pattern + [{"LOWER": "hp"}]]  # MotorStrength
motor_capacity_pattern = [numeric_qualifier_pattern + [{"LOWER": {"IN": ["cc", "l"]}}]]  # MotorCapacity

mileage_pattern = [numeric_qualifier_pattern + [{"LOWER": {"IN": ["miles", "km"]}}]]  # Mileage
price_pattern = [numeric_qualifier_pattern + [{"LOWER": {"IN": ["dollars", "$"]}}]]  # Price
drivetrain_pattern = [[{"LOWER": {"IN": ["fwd", "rwd", "awd"]}}]]  # Drivetrain
color_pattern = [[{"LOWER": {"IN": ["silver", "black", "red", "blue", "white"]}}]]  # Color

matcher.add("FUEL_TYPE", fuel_type_pattern)
matcher.add("TRANSMISSION", transmission_pattern)
matcher.add("BODY_STYLE", body_style_pattern)
matcher.add("MOTOR_STRENGTH", motor_strength_pattern)
matcher.add("MOTOR_CAPACITY", motor_capacity_pattern)
matcher.add("MILEAGE", mileage_pattern)
matcher.add("PRICE", price_pattern)
matcher.add("DRIVETRAIN", drivetrain_pattern)
matcher.add("COLOR", color_pattern)

def parse_detailed_car_query(query):
    doc = nlp(query)
    car_attributes = {
        "FuelType": [],
        "Transmission": [],
        "BodyStyle": [],
        "MotorStrength": [],
        "MotorCapacity": [],
        "Mileage": [],
        "Price": [],
        "Drivetrain": [],
        "Color": []
    }

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        span_text = span.text.lower()
        rule_id = nlp.vocab.strings[match_id]

        if rule_id == "FUEL_TYPE":
            car_attributes["FuelType"].append(span_text)
        elif rule_id == "TRANSMISSION":
            car_attributes["Transmission"].append(span_text)
        elif rule_id == "BODY_STYLE":
            car_attributes["BodyStyle"].append(span_text)
        elif rule_id == "DRIVETRAIN":
            car_attributes["Drivetrain"].append(span_text)
        elif rule_id == "COLOR":
            car_attributes["Color"].append(span_text)

        if rule_id == "MOTOR_CAPACITY":
            capacity_text = span.text.lower()
            if "l" in capacity_text or "liter" in capacity_text:
                num_value = [float(token.text) * 1000 for token in span if token.like_num]
                car_attributes["MotorCapacity"].extend(num_value)
            else:
                car_attributes["MotorCapacity"].extend([int(token.text) for token in span if token.like_num])

        if rule_id in ["MILEAGE", "MOTOR_STRENGTH", "PRICE"]:
            num_value = None
            comparison_qualifier = "specific" 
            for token in span:
                if token.like_num:
                    num_value = int(token.text)
                elif token.text in ["under", "less", "below","less_than"]:
                   comparison_qualifier = "less_than"
                elif token.text in ["over", "more", "above","more than"]:
                    comparison_qualifier = "greater_than"

            attribute = ''.join(word.capitalize() for word in rule_id.lower().split('_'))  # Convert rule_id to attribute name
            if num_value is not None:
               car_attributes[attribute].append(f"{num_value} {comparison_qualifier}")


    return car_attributes


query = "I'm looking for a diesel sedan, automatic, under 20000 dollars, less 30000 miles, over 200 hp, silver color,awd"
car_attributes = parse_detailed_car_query(query)
print(car_attributes)
