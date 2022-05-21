
import streamlit as st


import numpy as np
import pandas as pd
import nltk
import random
import os
from os import path
from PIL import Image

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS


# Pre-Processing
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from sklearn.utils import resample
from sklearn.utils import shuffle

# Modeling
import statsmodels.api as sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.util import ngrams
from collections import Counter
from gensim.models import word2vec


st.title("Sentiment Prediction Modelling")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file,encoding = 'unicode_escape')
    smpl = st.slider("Pick a value",0,23487,step = 1000)
    df1 = df1.sample(smpl)
    

# Delete missing observations for following variables
for x in ["Division Name","Department Name","Class Name","Review Text"]:
    df = df1[df1[x].notnull()]

# Extracting Missing Count and Unique Count by Column
unique_count = []
for x in df.columns:
    unique_count.append([x,len(df[x].unique()),df[x].isnull().sum()])

# Missing Values
print("Missing Values: {}".format(df.isnull().sum().sum()))

# Data Dimensions
print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))

# Create New Variables: 
# Word Length
df["Word Count"] = df['Review Text'].str.split().apply(len)
# Character Length
df["Character Count"] = df['Review Text'].apply(len)
# Boolean for Positive and Negative Reviews
df["Label"] = 0
df.loc[df.Rating >= 3,["Label"]] = 1


from nltk.stem.api import StemmerI
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
#ps = LancasterStemmer()
ps = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ') #1
    words = tokenizer.tokenize(txt) #2
    words = [w for w in words if not w in stop_words] #3
    #words = [ps.stem(w) for w in words] #4
    return words

# Pre-Processing
SIA = SentimentIntensityAnalyzer()
df["Review Text"]= df["Review Text"].astype(str)

# Applying Model, Variable Creation
df['Polarity Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['compound'])
df['Neutral Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['neu'])
df['Negative Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['neg'])
df['Positive Score']=df["Review Text"].apply(lambda x:SIA.polarity_scores(x)['pos'])

# Threshold
th = st.slider("Pick a value",0.0,1.0,step = 0.1)



# Converting 0 to 1 Decimal Score to a Categorical Variable
df['Sentiment']=''
df.loc[df['Polarity Score']>th,'Sentiment']='Positive'
df.loc[df['Polarity Score'].between(-th, +th),'Sentiment']='Neutral'
#df.loc[df['Polarity Score']==0.1,'Sentiment']='Neutral'
df.loc[df['Polarity Score']<-th,'Sentiment']='Negative'

df_pos=df[df["Sentiment"] == 'Positive']
df_neu=df[df["Sentiment"] == 'Neutral']
df_neg=df[df["Sentiment"] == 'Negative']

df_neu_upsampled = resample(df_neu, 
                                 replace=True,     # sample with replacement
                                 n_samples= df_pos.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
df_neg_upsampled = resample(df_neg, 
                                 replace=True,     # sample with replacement
                                 n_samples= df_pos.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
df = pd.concat([df_pos, df_neu_upsampled,df_neg_upsampled])

df['tokenized'] = df["Review Text"].astype(str).str.lower() # Turn into lower case text
df['tokenized'] = df.apply(lambda row: tokenizer.tokenize(row['tokenized']), axis=1) # Apply tokenize to each row
df['tokenized'] = df['tokenized'].apply(lambda x: [w for w in x if not w in stop_words]) # Remove stopwords from each row
df['tokenized'] = df['tokenized'].apply(lambda x: [ps.stem(w) for w in x]) # Apply stemming to each row
all_words = nltk.FreqDist(preprocessing(df['Review Text'])) # Calculate word occurrence from whole block of text

vocab_count = 200
word_features= list(all_words.keys())[:vocab_count] # 2000 most recurring unique words



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics


vect = TfidfVectorizer()
vect.fit(df["Review Text"])
X = vect.transform(df["Review Text"])

y = df["Sentiment"].copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=23, stratify=y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
lr=model.fit(X_train, y_train)
st.write("Train Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_train), y_train)))


#print("Validation Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_valid), y_valid)))
Test = st.text_input("Test")
Test = [Test]
textv= vect.transform(Test)




pred = lr.predict(textv)
st.write("The predicted Sentiment is",pred)