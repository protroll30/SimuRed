import os
from typing import cast

from dotenv import load_dotenv
from litellm import ModelResponse, acompletion

class LLMClient:
    def __init__(self):
        self.model = "gemini/gemini-3-flash-preview"
        self.api_key = os.getenv("GEMINI_API_KEY")

    async def get_responses(self, prompt_dict: dict):
        """
        Standard generator: Gets responses for all attack variations.
        """
        responses = {}
        for attack_type, text in prompt_dict.items():
            try:
                response = await acompletion(
                    model=self.model,
                    messages=[{"content": text, "role": "user"}],
                    api_key=self.api_key,
                    stream=False,
                )
                completion = cast(ModelResponse, response)
                content = completion.choices[0].message.content
                responses[attack_type] = content if content is not None else ""
            except Exception as e:
                responses[attack_type] = f"Error: {str(e)}"
        return responses

    async def judge_equivalence(self, original_out: str, mutated_out: str) -> bool:
        """
        The 'Logic Guard': Specifically looks for semantic drift or flipped meaning.
        """
        # Tight, specific prompt to minimize 'vibe' and maximize 'logic'
        judge_prompt = f"""
        INSTRUCTION: Compare these two AI responses. 
        Determine if they are LOGICALLY EQUIVALENT in substance.
        If they share the same conclusion/reasoning, answer 'YES'.
        If the meaning has changed, flipped, or drifted, answer 'NO'.
        
        RESPONSE A: {original_out}
        RESPONSE B: {mutated_out}
        
        ANSWER ONLY 'YES' OR 'NO'.
        """

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"content": judge_prompt, "role": "user"}],
                api_key=self.api_key,
                max_tokens=2,  # 1 word, cheap
                temperature=0.0,
                stream=False,
            )
            completion = cast(ModelResponse, response)
            raw = completion.choices[0].message.content or ""
            verdict = raw.strip().upper()
            return "YES" in verdict
            
        except Exception as e:
            print(f"⚖️ Judge Failed: {e}")
            return False # Default to 'Unstable' if we can't be sure