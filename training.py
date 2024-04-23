import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

lemmatizer= WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_up_sentence(sentence):
    # Tokenization
    sentence_words = word_tokenize(sentence)
    # Lowercasing and removing punctuation
    sentence_words = [re.sub(r'\W+', '', word.lower()) for word in sentence_words]
    # Remove stop words and extra whitespaces
    sentence_words = [word for word in sentence_words if word not in stop_words and word.strip() != '']
    # Lemmatization 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


intents=json.loads(open('intents_v3.json').read())

words=[]
classes=[]
documents=[]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Apply the cleaning and preprocessing steps
        cleaned_words = clean_up_sentence(pattern)
        words.extend(cleaned_words)
        documents.append((cleaned_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = sorted(set(words))
classes = sorted(set(classes))

classes=sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)


random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist=model.fit(trainX, trainY, epochs=1000, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print('Done')











