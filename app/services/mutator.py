import random

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk

class PromptMutator:
    def __init__(self, seed=42):
        random.seed(seed)
        # Ensure WordNet is available for synonym swapping
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        # 1. The Typo Attacker (Keyboard Noise)
        self.typo_aug = nac.KeyboardAug(
            aug_char_p=0.1,
            aug_word_p=0.1,
        )

        # 2. The Semantic Attacker (Synonym Swap)
        self.syn_aug = naw.SynonymAug(
            aug_src="wordnet",
            aug_p=0.2,
        )

    def generate_attacks(self, prompt: str) -> dict:
        """
        Generates three versions of the prompt: 
        Original, Fat-Finger Typos, and Synonym Swaps.
        """
        attacks = {
            "original": prompt,
            "typo_attack": self.typo_aug.augment(prompt)[0],
            "semantic_attack": self.syn_aug.augment(prompt)[0]
        }
        return attacks