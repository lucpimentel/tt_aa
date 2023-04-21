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
                                                max_tokens = max_tokens,
                    messages=[{"role": "system", "content": template},
                            {"role": "user", "content": prompt}]#f'Please create a tweet based on this note {prompt}:'}]
                            )
        return response['choices'][0]['message']['content']
