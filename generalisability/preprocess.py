import re
from nltk.corpus.reader import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def clean_text(text):
    stop_words = set(stopwords.words("english"))
    # english_stopwords = stopwords.words("english")
    english_stemmer = SnowballStemmer("english")
    text = text.replace('', '') # Remove 
    text = re.sub(r'[^\w]', ' ', text) # Remove symbols
    text = re.sub(r'[ ]{2,}', ' ', text) # Remove extra spaces
    text = re.sub(r'[ \t]+$', '', text) # Remove trailing white spaces
    tokens = []
    for token in text.split():
        if token not in stop_words:
            token = english_stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)

def preprocess_pipeline(text):
    return clean_text(text)

def vectorizer(text):
    count_vectorizer = pickle.load(open("vectorizers/count_vectorizer.pkl", "rb"))
    return count_vectorizer.transform(text)

