import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy
import pyttsx3
import speech_recognition as sr
import pyaudio
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


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
    global responses
    v = int(input("Please choose you language... English: 1 , Tamil:2: \n"))
    if (v == 1):
        s = int(input("Enter 1 for chat || Enter 2 for Voice : \n"))
        if (s == 1):
            print("Start talking with the bot (type quit to stop)!")
        elif (s == 2):
         while True:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print('How may I help:')
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                print('Done!')
            text = r.recognize_google(audio, language='en-IN')
            print(r.recognize_google(audio))
            q=r.recognize_google(audio)
            while True:
                inp = q
                if inp.lower() == "quit":
                    break

                results = model.predict([bag_of_words(inp, words)])[0]
                results_index = numpy.argmax(results)
                tag = labels[results_index]
                if results[results_index] > 0.7:
                    for tg in data["intents"]:
                        if tg['tag'] == tag:
                            responses = tg['responses']
                    print(random.choice(responses))
                else:
                    print("Sorry I didn't get that, Come again")
                break

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("Sorry I didn't get that, Come again")

    if (v == 1):
        s = int(input("Enter 1 for chat || Enter 2 for Voice : \n"))
        if (s == 1):
            print("Start talking with the bot (type quit to stop)!")
        elif (s == 2):
            while True:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    print('How may I help:')
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                    print('Done!')
                    text = r.recognize_google(audio, language='ta-IN')
                    print(r.recognize_google(audio))
                    q = r.recognize_google(audio)


chat()
