# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:38:58 2021

@author: Sandy
"""

import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import nltk
nltk.download('wordnet')

def welcome():
    return 'welcome all'

def main():
    st.title("Women clothing!")
    
    html_temp = """
    <div style = "background-color:yellow;padding:13px"
    <h1 style = "color:black;text-align:center;">Women's Clothing Prediction App </h1>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html= True)
    
    data = pd.read_csv(r'C:\Users\User\Documents\MERILYTICS\Sentiment Analysis\Streamlit\Womens Clothing E-Commerce Reviews.csv')
    data['Review_Text'].fillna('')
    
    def clean_and_tokenize(review):
        text = review.lower()
        
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokens = tokenizer.tokenize(text)
        
        stemmer = nltk.stem.WordNetLemmatizer()
        text = " ".join(stemmer.lemmatize(token) for token in tokens)
        text = re.sub("[^a-z']"," ", text)
        return text
    data["Review_Text"] = data["Review_Text"].astype(str)
    data["Clean_Review"] = data["Review_Text"].apply(clean_and_tokenize)
    
    #data = data[data['Rating'] !=3 ]
    #data['Sentiment'] = data['Rating'] >=4
    #data.head()
    
    
    
    tvect = TfidfVectorizer(min_df=1,max_df=1)
    
    X = tvect.fit_transform(data['Clean_Review'])
    y = data['Rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8,random_state = 0)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    Test = st.text_input("Test")
    Test = [Test]
    Test_Vector = tvect.transform(Test)
    
    result = ""
    
    if st.button("Predict"): 
        result = classifier.predict(Test_Vector)
        if result < 2:
            result = "Negative"
        elif result > 3:
            result = "Positive"
        else:
            result = "NA"
    st.success('The output is {}'.format(result)) 
     
if __name__=='__main__': 
    main() 
    