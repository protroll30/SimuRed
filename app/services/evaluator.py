from sentence_transformers import SentenceTransformer, util
import torch

class LogicEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes the Cosine Similarity between two text embeddings.
        1.0 = Identical Meaning | 0.0 = Completely Unrelated
        """
        # Encode text into vectors
        embeddings = self.model.encode([text_a, text_b], convert_to_tensor=True)
        
        # Calculate Cosine Similarity
        cosine_sim = util.cos_sim(embeddings[0], embeddings[1])
        
        return round(float(cosine_sim.item()), 4)

    def evaluate_drift(self, responses: dict) -> dict:
        original = responses.get("original", "")
        if not original:
            return {}

        scores = {}
        for attack_type, response_text in responses.items():
            if attack_type == "original":
                continue
            
            similarity = self.calculate_semantic_similarity(original, response_text)
            
            # THE THRESHOLD:
            # 0.85 is a safe "Stable" threshold for professional LLM work.
            scores[attack_type] = {
                "similarity_score": similarity,
                "is_stable": similarity > 0.85
            }
            
        return scores