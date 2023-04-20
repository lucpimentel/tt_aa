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

os.environ['OPENAI_API_KEY'] = api_key


user_prompt = st.text_input('Please insert your note:')

openai_embeddings = OpenAIEmbeddings()
db = FAISS.load_local('faiss_index',openai_embeddings)


button_isClicked = st.button("Generate tweet",)


# Create a button to call the function
if button_isClicked:
    with st.spinner("Loading..."):

        references = [tweet.page_content for tweet in db.similarity_search(user_prompt,k=3)]

        template = f'''You are a world-class twitter ghostwriter. Your goal is to create a tweet for me.
        It must be precise in its wording, be inspirational, and sound like I am speaking from my own experience.

        For reference, use these high-performing tweets as templates.

        Template 1: {references[0]}

        Template 2: {references[1]}

        Template 3: {references[2]}
        '''

        response_text = openai_api_call(user_prompt, template)
        st.text_area('Tweet:',response_text,height = 200)
        button_isClicked = False
