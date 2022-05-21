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

st.title("Predicting the Sentiment")
st.subheader("Instructions")
st.text("Enter The Text in the Textbox, select the Model.")
st.sidebar.subheader("Development of A Sentiment Analysis App")
st.sidebar.text("For Predicting the Sentiment")
st.sidebar.info("Using 4 Different Models")


        # Sentiment Analysis
if st.text("Let's Start the App"):
		#st.subheader("Analyse Your Text")

    text = st.text_area("Enter Text","Type Here ..")
    #if st.button("Predict the Sentiment"):
        #st.sidebar.markdown("### Sentiment Prediction")
    select = st.sidebar.selectbox('Select the Model and check the Show box',['TextBlob','Flair','Vader','Stanza'])

    if st.sidebar.checkbox('Show',False,key='0'):
            st.markdown("### Sentiment Prediction ")
            if select=='TextBlob':
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
            elif select=='Flair':
                # Flair Model
                fls = flair.data.Sentence(text)
                flair_sentiment.predict(fls)
                topSentiment = fls.labels[0].value
                score = fls.labels[0].score
                #st.success(score)

                if score < 0.8:
                    st.write("Flair Model Prediction is Neutral")
                elif topSentiment =='POSITIVE':
                    st.write("Flair Model Prediction is Positive")
                elif topSentiment == 'NEGATIVE' :
                    st.write("Flair Model Prediction is Negative")
                #st.write(topSentiment)
                #st.write ("Flair Model Prediction is", topSentiment)
            elif select=='Vader':
            # Vader Model
                polarityDict = sid.polarity_scores(text)
                compound = polarityDict['compound']
                if compound < 0.4:
                    st.write ("Vader Model Prediction is Negative")
                elif compound > 0.6:
                    st.write ("Vader Model Prediction is Positive")
                else:
                    st.write ("Vader Model Prediction is Neutral")
            else:
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


#if __name__ == '__main__':
#	main()
