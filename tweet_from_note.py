import streamlit as st
import openai
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from auxfunctions import openai_api_call
import os
import pandas as pd






st.title('Welcome to the GPT Tweet From Note App!')

api_key = st.text_input('What is your OpenAI API key?')

#os.environ['OPENAI_API_KEY'] = api_key


user_prompt = st.text_input('Please insert your note:')

k = st.slider('Select the number of relevant examples:', 1, 10, 2, step = 1)

temperature = st.slider("Select a temperature value", 0.0, 1.0, 0.6, step=0.05)

top_p = st.slider("Select a top_p value", 0.0, 1.0, 1.0, step=0.05)

try:
    openai_embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    db = FAISS.load_local('faiss_index',openai_embeddings)
except:
    pass

button_isClicked = st.button("Generate tweet",)


# Create a button to call the function
if button_isClicked:
    with st.spinner("Loading..."):

        references = [tweet.page_content for tweet in db.similarity_search(user_prompt,k=k)]

        template = f'''You are the best tweet writer in the world.
            Your goal is to turn one of my personal notes into a highly insightful tweet.
            
            Use the structure and style of the tweets down below:

            
            '''

        for i in range(k):
            template += f'\nTemplate {i+1}: {references[i]}\n'

        #template += '\nSo now that you are armed with all of the necessary information, I will give you one of my personal notes for you to create 1 tweet.'

        #st.write(template)

        response_text = openai_api_call(user_prompt, template, temperature = temperature, top_p = top_p)
        st.text_area('Tweet:',response_text,height = 200)
        st.write(references)
        button_isClicked = False
