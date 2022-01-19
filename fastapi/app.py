# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from InputMsgs import InputMsg
#import numpy as np
#import pickle
import pandas as pd
import joblib


import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import os
print(os.getcwd())

# 2. Create the app object
app = FastAPI()

pkl_filename = "naivebayesmodel.pkl"
model = joblib.load(pkl_filename)

pkl_filename = "count_vectorizer.pkl"
vectorizer = joblib.load(pkl_filename)

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    #corpus = []

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    #corpus.append(review)
    return review


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Welcome from the API'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_sms(data:InputMsg):
    data = data.dict()
    msg=data['msg']
   
    sms = pd.DataFrame([msg],columns=["v2"])
    Y=sms['v2'].apply(clean_text)
    Y=vectorizer.transform(Y).toarray()
    prediction = model.predict(Y)
    if(prediction[0] == 1):
        prediction="SPAM"
    else:
        prediction="HAM"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload