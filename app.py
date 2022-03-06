from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
 
from nltk.tokenize import word_tokenize
import joblib

def preprocess(t):
     text=t
     text = text.lower()
     text = re.sub('\[.*?\]', '', text) # remove square brackets
     text = re.sub(r'[^\w\s]','',text) # remove punctuation
     text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
     text = re.sub('\n', '', text)
     stop_words = stopwords.words('english')
     text=" ".join([word for word in text.split() if word not in stop_words])

     text=word_tokenize(text)
     
     
     return text

tfidf = joblib.load('TFIDF')
model = joblib.load('model')


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message = request.form['text']
        message = preprocess(message)
        
        vect = tfidf.transform(message).toarray()
        prediction = model.predict(vect)
        
	  
    return render_template('result.html',pred = prediction,msg=message)






if __name__=='__main__':
    app.run(debug=True)
