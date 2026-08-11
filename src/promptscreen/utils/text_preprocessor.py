"""Shared text pre-processing utilities for ML-based guards."""

import string
from typing import Optional, cast

import emoji
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from .unicode_normalize import normalize_for_model


class TextPreProcessor:
    """Normalises raw prompt text for ML classifiers.

    Steps applied:
    0. Unicode normalization (undoes zero-width/tag/combining-mark
       obfuscation so real words survive tokenization -- see
       unicode_normalize.normalize_for_model)
    1. Lower-case
    2. Emoji demojization (e.g. 😀 → :grinning_face:)
    3. Punctuation removal
    4. Tokenization via NLTK word_tokenize
    5. Stop-word removal
    6. POS-aware lemmatization
    """

    def __init__(self, custom_stopwords: Optional[set] = None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

    def _get_wordnet_pos(self, word: str) -> str:
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return cast(str, tag_dict.get(tag, wordnet.NOUN))

    def preprocess(self, text: str) -> str:
        text = normalize_for_model(text)
        text = text.lower()
        text = emoji.demojize(text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        filtered_tokens = [
            self.lemmatizer.lemmatize(token, self._get_wordnet_pos(token))
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        return " ".join(filtered_tokens)
