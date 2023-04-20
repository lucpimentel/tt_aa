import openai
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

def openai_api_call(prompt: str, template: str, model = 'gpt-3.5-turbo', temperature: int = 0.5, top_p:int = 0.5, max_tokens:int = 1000) -> str:
        """
        Calls the OpenAI API with a given prompt

        Args:
            prompt (str): The prompt to use for generating the text.
        
        Returns:
            str: The generated text.
        """
        # Create the completion call using the OpenAI API
        
        
        response = openai.ChatCompletion.create(model=model,
                                                temperature = temperature,
                                                top_p = top_p,
                    messages=[{"role": "system", "content": template},
                            {"role": "user", "content": f'Please create a tweet based on this note {prompt}:'}]
                            )
        return response['choices'][0]['message']['content']
        
        


        # Extract and return the generated text from the completion call response
        response_text = completion_dict['choices'][0]['text'].strip()
        return response_text



def write_tweet(user_prompt: str, df: pd.DataFrame) -> str:


    def get_embedded_vector(text: str, sleep:bool = True) -> list[float]:
        """
        Returns the embedded vector for a given text string using OpenAI API.
        
        Args:
            text (str): Input text to be embedded.
        
        Returns:
            list[float]: Embedded vector for the input text.
        """
        # Use OpenAI API to generate embedded vector for input text
        embedding_dict = openai.Embedding.create(
            input = [text],
            model = 'text-embedding-ada-002'
        )
        
        # Extract embedded vector from the API response
        embedded_vector = embedding_dict['data'][0]['embedding']

        return embedded_vector
    
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Returns the cosine similarity of two vectors.
        
        Args:
            v1 (np.ndarray): A numpy array representing the first vector.
            v2 (np.ndarray): A numpy array representing the second vector.
        
        Returns:
            float: The cosine similarity of the two vectors.
        """
        # Compute the cosine similarity of the two vectors using the cosine distance formula
        cosine_sim = 1 - cosine(v1, v2)
        
        return cosine_sim

    def create_final_prompt(reference,user_prompt):

        final_prompt = f'''
        You are a world-class twitter ghostwriter. Your goal is to create a tweet for me.
        It must be precise in its wording, be inspirational, and sound like I am speaking from my own experience.

        For reference, use these high-performing tweets as templates.

        Template 1: {reference[0]}

        Template 2: {reference[1]}

        Template 3: {reference[2]}

        So, now that you are armed with all of the relevant information, please create a tweet based on this note "{user_prompt}":'''

        return final_prompt

    def generate_tweet(prompt: str, temperature: int = 0.5, top_p:int = 0.5, max_tokens:int = 1000) -> str:
        """
        Generates text based on the given prompt using the default OpenAI model (text-davinci-002)
        and temperature (0.5).
        
        Args:
            prompt (str): The prompt to use for generating the text.
        
        Returns:
            str: The generated text.
        """
        # Create the completion call using the OpenAI API
        completion_dict = openai.Completion.create(
            prompt=prompt,
            model="gpt-3.5-turbo",
            temperature=temperature,
            max_tokens= max_tokens,
            top_p = top_p
        )
        
        # Extract and return the generated text from the completion call response
        generated_text = completion_dict['choices'][0]['text'].strip()
        return generated_text



    # Get the embedded vector for the user prompt
    user_prompt_vector = get_embedded_vector(user_prompt, sleep=False)
    
    
    # Find the cosine similarity between user_prompt_vector and email_content_vector for each row in the dataframe
    df = df.assign(similarity=lambda x: x['embedded_vector'].apply(lambda x: cosine_similarity(user_prompt_vector, x)))
    
    # Get the indices of the top 3 most similar email contents
    top_n_indices = df['similarity'].nlargest(3).index
    
    # Create a dictionary of the most similar email contents
    most_similar_contents_dict = df.loc[top_n_indices].reset_index()['text_content'].to_dict()

    # Use the most similar email contents to engineer the final prompt
    engineered_prompt = create_final_prompt(most_similar_contents_dict, user_prompt)
    
    # Generate short copy using the engineered prompt
    tweet = generate_tweet(engineered_prompt, 0.75, 1)
    
    return tweet