from sentence_transformers import SentenceTransformer, util
import torch
import re

class LogicEvaluator:
    def __init__(self, llm_client):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.llm = llm_client 
        self.negation_words = {'not', 'no', 'never', 'none', 'cannot', "n't"}

    def calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        The Fast Pass: Measures how close two strings are in vector space.
        """
        embeddings = self.model.encode([text_a, text_b], convert_to_tensor=True)
        cosine_sim = util.cos_sim(embeddings[0], embeddings[1])
        return round(float(cosine_sim.item()), 4)

    def _contains_logical_flip(self, text_a: str, text_b: str) -> bool:
        """
        Quick check for negation words to catch 'False Positives' 
        (High similarity score but opposite meaning).
        """
        words_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w+\b', text_b.lower()))
        # Returns True if one has a 'not/no' and the other doesn't
        return not (self.negation_words & words_a == self.negation_words & words_b)

    async def evaluate_drift(self, responses: dict) -> dict:
        """
        The Cascaded Judge: Uses math for speed, LLM for logic.
        """
        original = responses.get("original", "")
        if not original: return {}

        scores = {}
        for attack_type, response_text in responses.items():
            if attack_type == "original": continue
            
            # Step 1: Fast Embedding Check
            similarity = self.calculate_semantic_similarity(original, response_text)
            
            # Step 2: Determine if we need to spend money on a Gemini Judge
            # We judge if it's in the 'Gray Zone' OR if a negation flip is suspected
            needs_judge = (0.60 < similarity < 0.88) or (similarity > 0.88 and self._contains_logical_flip(original, response_text))
            
            # Default stability based on 0.70 threshold
            is_stable = similarity >= 0.70 
            
            # Step 3: Final Logical Verdict (The 'Judge' override)
            if needs_judge:
                is_stable = await self.llm.judge_equivalence(original, response_text)
            
            scores[attack_type] = {
                "similarity_score": similarity,
                "is_stable": is_stable,
                "method": "Gemini Judge" if needs_judge else "Vector Math"
            }
            
        return scores