import streamlit as st
import pickle
import pandas as pd
import string
import nltk
import nltk
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


tfidf=pickle.load(open('text_transformer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def text_pre_process(text):
    sample_list=[]
    filtered_sentence=[]
    final_preprocessed_list=[]
    text=text.lower()
    text=word_tokenize(text)
    for i in text:
        if i.isalnum():
                sample_list.append(i)
    for w in sample_list:
        if w not in stop_words and w not in string.punctuation:
            filtered_sentence.append(w)
    for i in filtered_sentence:
        final_preprocessed_list.append(ps.stem(i))

    return " ".join(final_preprocessed_list)


st.title('SMS/EMAIL SPAM CLASSIFIER APPLICATION')

#Input section
sms_text=st.text_area('Enter the text message')

#Steps to follow:
#Pre-processing step
#Vectorization
#Prediction
#Display

if st.button('Predict'):

    # preprocess
    transformed_sms = text_pre_process(sms_text)
    #vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

