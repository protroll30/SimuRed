import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

class PromptMutator:
    def __init__(self):
        # 1. The Typo Attacker (Keyboard Noise)
        # This simulates a user with 'fat fingers'.
        self.typo_aug = nac.KeyboardAug(aug_char_p = 0.1, aug_word_p = 0.1)
        # 2. The Semantic Attacker (Synonym Swap)
        # This uses WordNet (a massive dictionary) to swap words with synonyms.
        self.syn_aug = naw.SynonymAug(aug_src = 'wordnet', aug_p = 0.2)

    def generate_attacks(self, prompt: str) -> dict:
        attacks = {
            "original": prompt,
            "typo_attack": self.typo_aug.augment(prompt)[0],
            "semantic_attack": self.syn_aug.augment(prompt)[0]
        }
        return attacks



