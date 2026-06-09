import random
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.bigram_counts = defaultdict(list)
        self.build_model(corpus)

    def tokenize(self, text):
        return text.lower().replace(".", "").replace(",", "").split()

    def build_model(self, corpus):
        for sentence in corpus:
            words = self.tokenize(sentence)

            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                self.bigram_counts[current_word].append(next_word)

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            possible_next_words = self.bigram_counts.get(current_word)

            if not possible_next_words:
                break

            next_word = random.choice(possible_next_words)
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)