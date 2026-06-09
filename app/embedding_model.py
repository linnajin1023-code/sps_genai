import spacy

nlp = spacy.load("en_core_web_md")


def get_word_embedding(word: str):
    doc = nlp(word)

    if len(doc) == 0:
        return None

    token = doc[0]
    return token.vector.tolist()