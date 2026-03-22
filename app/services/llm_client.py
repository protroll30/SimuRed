import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.model = "gemini/gemini-1.5-flash"
        self.api_key = os.getenv("GEMINI_API_KEY")

    async def get_responses(self, prompt_dict: dict):
        """
        Takes the dictionary from the Mutator and gets AI answers for all of them.
        """
        responses = {}
        
        for attack_type, text in prompt_dict.items():
            # We use litellm.completion to get the answer
            response = await completion(
                model=self.model,
                messages=[{"content": text, "role": "user"}],
                api_key=self.api_key
            )
            # Extract just the text content from the AI
            responses[attack_type] = response.choices[0].message.content
            
        return responses