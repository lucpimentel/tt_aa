import streamlit as st
import openai
import pandas as pd
from auxfunctions import write_tweet
import os
import pandas as pd



st.title('Welcome to the GPT Tweet From Note App!')

api_key = st.text_input('What is your OpenAI API key?')

openai.api_key = api_key


user_prompt = st.text_input('Please insert your note:')

#df = pd.read_json('Tweet swipe file - embedded vectors.json')
# read all json files in the same folder, turn them into a dataframe, and concat everything
# Get the current working directory
cwd = os.getcwd()

# Create an empty list to hold the dataframes
df_list = []

# Loop through all files in the directory
for file in os.listdir(cwd):
    if file.endswith('.json'):
        # Read the JSON file into a dataframe
        df = pd.read_json(os.path.join(cwd, file))
        # Append the dataframe to the list
        df_list.append(df)

# Concatenate all dataframes in the list into a single dataframe
concatenated_df = pd.concat(df_list, ignore_index=True)



# streamlit with dropdown of two options: serious tone = st, funny
tone = st.selectbox('What tone of voice would you like to use?',('serious','conversational','informal','casual','assertive','intriguing','formal'))

length = st.selectbox('How long would you like the tweet to be?',('short','medium','long'))

#prompt_1 = f'''"{user_prompt}" Please turn this note into a {length} tweet.
# It must have a {tone} tone of voice, be precise in its wording, be inspirational, and sound like I am speaking from my own experience.'''



button_isClicked = st.button("Generate tweet",)


# Create a button to call the function
if button_isClicked:
    with st.spinner("Loading..."):
        response_text = write_tweet(user_prompt,concatenated_df)
        st.text_area('Tweet:',response_text,height = 200)
        button_isClicked = False
