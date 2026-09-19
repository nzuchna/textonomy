import re
from typing import List

import nltk
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")


class TextPreprocessor:
    def __init__(self, additional_stopwords: List[str] = None):
        self.stop_words = set(stopwords.words("english"))
        if additional_stopwords:
            self.stop_words.update(additional_stopwords)
        self.lemmatizer = WordNetLemmatizer()
        self.bigram_model = None

    def clean_text(self, text: str) -> str:
        text = text.lower()
        # text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove special characters and numbers
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return simple_preprocess(text, deacc=True)

    def process_text(
        self,
        text: str,
        clean_html: bool = True,
        clean: bool = True,
        tokenize: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        use_bigrams: bool = True,
    ) -> List[str] | str:
        """If tokenize is False, all set parameters after tokenize will be ignored."""
        if clean_html:
            text = self.clean_html(text)

        if clean:
            text = self.clean_text(text)

        if not tokenize:
            return text

        tokens = self.tokenize(text) if tokenize else text.split()

        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        if lemmatize:
            tokens = self.lemmatize(tokens)

        if use_bigrams and self.bigram_model:
            tokens = self.bigram_model[tokens]

        return tokens

    def process_word(
        self,
        word: str,
        clean_html: bool = True,
        clean: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ) -> str:
        if clean_html:
            word = self.clean_html(word)
        if clean:
            word = self.clean_text(word)
        tokens = [word]
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        if lemmatize:
            tokens = self.lemmatize(tokens)
        if tokens:
            return tokens[0]
        else:
            return ""

    def process_docs(
        self,
        docs: List[str],
        clean_html: bool = True,
        clean: bool = True,
        tokenize: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        use_bigrams: bool = True,
    ) -> List[List[str]]:
        processed_docs = [
            self.process_text(
                doc, clean_html, clean, tokenize, remove_stopwords, lemmatize, use_bigrams
            )
            for doc in docs
        ]
        return processed_docs

    def process_topic_words(
        self,
        topic_words: List[List[str]],
        clean: bool = False,
        clean_html: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ) -> List[List[str]]:
        topic_words = [
            [
                self.process_word(
                    word,
                    clean=clean,
                    clean_html=clean_html,
                    remove_stopwords=remove_stopwords,
                    lemmatize=lemmatize,
                )
                for word in topic
            ]
            for topic in topic_words
        ]
        return [[string for string in topic if string] for topic in topic_words]

    def train_bigram_model(
        self, docs: List[List[str]], min_count: int = 5, threshold: float = 10.0
    ):
        phrases = Phrases(docs, min_count=min_count)
        self.bigram_model = Phraser(phrases)


# Example Usage
if __name__ == "__main__":
    sample_texts = [
        "<p>Natural Language Processing is a fascinating field of Artificial Intelligence!</p>",
        "<div>Text preprocessing &amp; cleaning is crucial for topic modeling.</div>",
    ]

    preprocessor = TextPreprocessor(additional_stopwords=["artificial", "intelligence"])

    processed_docs = [preprocessor.process_text(text) for text in sample_texts]

    preprocessor.train_bigram_model(processed_docs, min_count=3)

    print("Processed with bigrams:")
    for doc in preprocessor.process_docs(sample_texts):
        print(doc)
