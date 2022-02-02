import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
from wordcloud import WordCloud
from collections import Counter
import csv
from matplotlib import rcParams
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

from flask import Flask, render_template, request
import pickle
import numpy as np

loaded_model = joblib.load('finalized_model.sav')
vectorizer = joblib.load('vectorizer.sav')

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('template.html')


@app.route('/', methods=['POST'])
def home():
    news = request.form['news']
    news = [news]
    news = vectorizer.transform(news)
    predict = loaded_model.predict(news)
    return render_template('after.html', data=predict)


if __name__ == "__main__":
    app.run(port= 3000, debug=True)