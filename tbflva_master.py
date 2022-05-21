from textblob import TextBlob
import streamlit as st
import flair
import stanza
from flair.models import TextClassifier
from flair.data import Sentence
flair_sentiment = flair.models.TextClassifier.load('en-sentiment');
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


#def main():
	#""" Sentiment Analysis with Streamlit """
    
st.title("Sentiment Analysis")
st.subheader("Predicting the Sentiment")

        # Sentiment Analysis
if st.text("Let's Start the App"):
		#st.subheader("Analyse Your Text")

    text = st.text_area("Enter Text","Type Here ..")
    if st.button("Predict the Sentiment"):
        
# TextBlob Model       
        tbp=TextBlob(text)
        s=tbp.sentiment
        st.success(s)
        p=tbp.sentiment.polarity
        if p < 0:
            st.write ("TextBlob Model Prediction is Negative")
        elif p > 0:
            st.write ("TextBlob Model Prediction is Positive")
        else:
            st.write ("TextBlob Model Prediction is Neutral")


# Stanza Model       
        nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
        doc = nlp(text)
        for i, sentence in enumerate(doc.sentences):(sentence.sentiment)
        stanza_pred=sentence.sentiment

        if stanza_pred == 0:
            st.write ("Stanza Model Prediction is Negative")
        elif stanza_pred == 2:
            st.write ("Stanza Model Prediction is Positive")
        else:
            st.write ("Stanza Model Prediction is Neutral")




# Flair Model            
        fls = flair.data.Sentence(text)
        flair_sentiment.predict(fls)
        topSentiment = fls.labels[0].value
        score = fls.labels[0].score
        st.success(score)
        
        if score < 0.8:
            st.write("Flair Model Prediction is Neutral")   
        elif topSentiment =='POSITIVE':
            st.write("Flair Model Prediction is Positive")
        elif topSentiment == 'NEGATIVE' :
            st.write("Flair Model Prediction is Negative")
            #st.write(topSentiment)
            #st.write ("Flair Model Prediction is", topSentiment)

# Vader Model
        polarityDict = sid.polarity_scores(text)
        compound = polarityDict['compound']
        if compound < 0.4:
            st.write ("Vader Model Prediction is Negative")
        elif compound > 0.6:
            st.write ("Vader Model Prediction is Positive")
        else:
            st.write ("Vader Model Prediction is Neutral")


            
    st.sidebar.subheader("Development of A Sentiment Analysis App")
    st.sidebar.text("For Predicting the Sentiment")
    st.sidebar.info("Using TextBlob")
    st.sidebar.subheader("Instructions")
    st.sidebar.text("Enter The Text in the Textbox")
    st.sidebar.text("The Predicted Sentiment will be")
    st.sidebar.text("Positive OR Neutral OR Negative")

#if __name__ == '__main__':
#	main()
