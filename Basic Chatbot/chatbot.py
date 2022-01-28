import nltk
import numpy
import random
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from keras.models import load_model

model = load_model('chatbot.h5')

import json
with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit" or inp.lower() == "bye" or inp.lower() == "good bye":
            break
        check = bag_of_words(inp, words)
        check1 = numpy.expand_dims(check,axis=0)
        results = model.predict(check1)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()