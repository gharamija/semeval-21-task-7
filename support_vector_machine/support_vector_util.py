from gensim.models import Word2Vec

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np


class TextPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.lower() not in self.stop_words]

        return ' '.join(tokens)


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    n_words = 0
    for word in words:
        if word in vocabulary:
            n_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if n_words:
        feature_vector = np.divide(feature_vector, n_words)
    return feature_vector


class Word2VecEncoder:
    def __init__(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        self.model = Word2Vec(sentences.apply(nltk.tokenize.word_tokenize), vector_size=vector_size, window=window,
                              min_count=min_count, workers=workers)

    def encode(self, sentence, num_features):
        vocabulary = set(self.model.wv.index_to_key)
        return np.vstack(
            [average_word_vectors(tokens, self.model, vocabulary, num_features) for tokens in sentence])
