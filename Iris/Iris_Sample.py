import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 
  
def welcome(): 
    return 'welcome all'
  
def prediction(sepal_length, sepal_width, petal_length, petal_width):   
   
    prediction = classifier.predict( 
        [[sepal_length, sepal_width, petal_length, petal_width]]) 
    print(prediction) 
    return prediction 
    
def main():  
    st.title("Iris Flower Prediction") 
      
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1> 
    </div> 
    """
 
    st.markdown(html_temp, unsafe_allow_html = True) 

    sepal_length = st.text_input("Sepal Length", "Type Here") 
    sepal_width = st.text_input("Sepal Width", "Type Here") 
    petal_length = st.text_input("Petal Length", "Type Here") 
    petal_width = st.text_input("Petal Width", "Type Here") 
    result =""  
    if st.button("Predict"): 
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
        if result == 0:
            result = "Iris-setosa"
            if result == 1:
                    result = "Iris-versicolor"
                    if result == 2:
                        result = "Iris-virginica"
                    else:
                        result = "NA"
    st.success('The output is {}'.format(result)) 
     
if __name__=='__main__': 
    main() 