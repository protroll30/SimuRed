import random

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nltk

class PromptMutator:
    def __init__(self, seed=42):
        random.seed(seed)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        self.typo_aug = nac.KeyboardAug(
            aug_char_p=0.1,
            aug_word_p=0.1,
        )

        self.syn_aug = naw.SynonymAug(
            aug_src="wordnet",
            aug_p=0.2,
        )

    def generate_attacks(self, prompt: str) -> dict:
        attacks = {
            "original": prompt,
            "typo_attack": self.typo_aug.augment(prompt)[0],
            "semantic_attack": self.syn_aug.augment(prompt)[0]
        }
        return attacks