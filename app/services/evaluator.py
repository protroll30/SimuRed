import re

class LogicEvaluator:
    def __init__(self):
        # 'clean' the text for a fair comparison
        self.clean_pattern = re.compile(r'[^a-zA-Z\s]')

    def _tokenize(self, text: str) -> set:
        """
        Cleans the text and turns it into a set of unique words.
        """
        clean_text = self.clean_pattern.sub('', text.lower())
        return set(clean_text.split())
    
    def calculate_jaccard(self, text_a: str, text_b: str) -> float:
        """
        Jaccard Similarity = (A ∩ B) / (A ∪ B)
        """
        set_a = self._tokenize(text_a)
        set_b = self._tokenize(text_b)
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        # Avoid division by zero
        if union == 0:
            return 1.0
            
        return round(intersection / union, 4)

    def evaluate_drift(self, responses: dict) -> dict:
        """
        Compares the 'attack' responses against the 'original' response.
        """
        original = responses.get("original", "")
        scores = {}

        for attack_type, response_text in responses.items():
            if attack_type == "original":
                continue
            
            # Calculate how much the logic drifted from the original
            similarity = self.calculate_jaccard(original, response_text)
            scores[attack_type] = {
                "similarity_score": similarity,
                "is_stable": similarity > 0.7  # If >70% same, we call it 'Stable'
            }
            
        return scores