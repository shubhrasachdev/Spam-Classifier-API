import pandas as pd
import re
import os
import string
import joblib
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import nltk

# Stopwords
APP = os.path.abspath(__file__)
FILE_DIR, APP_NAME = os.path.split(APP)
NLTK_DATA_PATH = os.path.join(FILE_DIR, 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

clf = joblib.load('classifier.pkl')
tfidf = joblib.load('tfidf.pkl')

def cleaning_text(text):
    return re.sub('[^a-zA-Z]', ' ', text).lower()
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
def tokenize_text(text):
    tokenized_text = text.split()
    return tokenized_text
def lemmatize_text(token_list):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        args = parser.parse_args()
        text = args['query']
        puncs = count_punct(text)
        clean_text = cleaning_text(text)
        text_len = len(text) - text.count(" ")
        tokens = tokenize_text(clean_text)
        lemmatized_text = lemmatize_text(tokens)
        vectorized_text = tfidf.transform([lemmatized_text])
        to_predict = pd.concat([pd.DataFrame([[text_len, puncs]], columns = ["text_len", "punct"]), 
                        pd.DataFrame(vectorized_text.toarray())] , axis = 1)
        prediction = clf.predict(to_predict)[0]
        # Output either 'Ham' or 'Spam' along with the score
        if prediction == 0:
            pred_text = 'Ham'
        else:
            pred_text = 'Spam'
            
        # round the predict proba value and set to new variable
        confidence = clf.predict_proba(to_predict)[0][prediction]

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output

api.add_resource(PredictSentiment, '/')
if __name__ == '__main__':
    app.run(debug=True)