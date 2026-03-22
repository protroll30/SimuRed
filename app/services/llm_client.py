import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        # We add '-latest' to ensure it maps correctly to the v1beta endpoint
        self.model = "gemini/gemini-1.5-flash-latest"
        self.api_key = os.getenv("GEMINI_API_KEY")

    async def get_responses(self, prompt_dict: dict):
        """
        Takes the dictionary from the Mutator and gets AI answers for all of them.
        """
        responses = {}
        
        for attack_type, text in prompt_dict.items():
            try:
                # We call the model using the GEMINI_API_KEY from .env
                response = await completion(
                    model=self.model,
                    messages=[{"content": text, "role": "user"}],
                    api_key=self.api_key
                )
                # Extract the text content
                responses[attack_type] = response.choices[0].message.content
            except Exception as e:
                # Fallback if one specific call fails
                responses[attack_type] = f"Error calling LLM: {str(e)}"
            
        return responses