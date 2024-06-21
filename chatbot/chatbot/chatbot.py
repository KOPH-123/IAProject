import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from flask import Flask, request, jsonify, render_template

import datetime
from googlesearch import *
import webbrowser
import requests
from pycricbuzz import Cricbuzz
import billboard
import time
from pygame import mixer
import COVID19Py
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

# Cargar los datos y el modelo del chatbot
intents = json.loads(open('C:\\Users\\hande\\OneDrive\\Escritorio\\semestre 2024\\IA\\python project\\chatbot\\chatbot\\intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

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

def get_response(return_list,intents_json):
    
    if len(return_list) == 0:
        tag = 'noanswer'
    else:    
        tag = return_list[0]['intent']
    
    response = ""

    if tag == 'datetime':        
        response = f"Hoy es {time.strftime('%A')}, {time.strftime('%d %B %Y')}, y la hora actual es {time.strftime('%H:%M:%S')}."
    
    elif tag == 'google':
        response = "You can ask me to search something on Google by specifying your query."
    
    elif tag == 'weather':
        api_key = '987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = "managua"  # Replace with dynamic input if needed
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()
        response = f"La temperatura de hoy en managua es: {round(x['main']['temp']-273, 2)}°C\nFeels Like: {round(x['main']['feels_like']-273, 2)}°C\n{x['weather'][0]['main']}"

    elif tag == 'news':
        main_url = "http://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = [] 
          
        for ar in article: 
            results.append([ar["title"], ar["url"]]) 
          
        response = ""
        for i in range(10): 
            response += f"{i + 1}. {results[i][0]}\n{results[i][1]}\n\n"
            
    elif tag == 'cricket':
        c = Cricbuzz()
        matches = c.matches()
        response = ""
        for match in matches:
            response += f"{match['srs']} {match['mnum']} {match['status']}\n"
    
    elif tag == 'song':
        chart = billboard.ChartData('hot-100')
        response = 'The top 10 songs at the moment are:\n'
        for i in range(10):
            song = chart[i]
            response += f"{i+1}. {song.title} - {song.artist}\n"

    elif tag == 'timer':        
        response = "Please provide the number of minutes for the timer."

    elif tag == 'covid19':
        response = "Please provide the name of the location you want the COVID-19 stats for."
    
    else:
        list_of_intents = intents_json['intents']    
        for i in list_of_intents:
            if tag == i['tag']:
                response = random.choice(i['responses'])
                
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    userText = request.json.get("message")
    ints = predict_class(userText)
    res = get_response(ints, intents)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)
