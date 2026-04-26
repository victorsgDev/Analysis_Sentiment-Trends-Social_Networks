import pandas as pd
import re
import os

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import requests
from dotenv import load_dotenv
load_dotenv()

import spacy
from collections import Counter
import math
from spacytextblob.spacytextblob import SpacyTextBlob


class DataExtractor:
    def __init__(self, source_file: str = None, chunksize: int = 100000):
        """
        Inicializa el extractor con el archivo de origen.

        Parámetros:
        - source_file: ruta al archivo CSV o JSON.
        - chunksize: tamaño de bloque para leer archivos grandes.
        """
        self.source_file = source_file
        self.data = None
        self.chunksize = chunksize

    def load_data(self):
        """
        Carga los datos del archivo de origen y los almacena en self.data.
        """
        if not self.source_file:
            raise ValueError("No source file defined.")

        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"File {self.source_file} not found.")

        if self.source_file.endswith(".csv"):
            reader = pd.read_csv(self.source_file, encoding="utf-8", engine="python", chunksize=self.chunksize, on_bad_lines='skip')
            self.data = pd.concat(reader, ignore_index=True)
        elif self.source_file.endswith(".json"):
            self.data = pd.read_json(self.source_file, encoding="utf-8", lines=True)
        else:
            raise ValueError("Unsupported file format. Please use CSV or JSON.")

        return self.data

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpia y normaliza un texto.
        """
        # validar texto
        if not isinstance(text, str):
            return ""
        # pasar a minúsculas
        text = text.lower()
        # quitar urls
        text = re.sub(r'http\S+|www\S+', "", text)
        # quitar menciones
        text = re.sub(r'@\w+', "", text)
        # quitar caracteres especiales excepto hashtags
        text = re.sub(r'[^\w\s#]', "", text)
        # compactar espacios
        text = re.sub(r'\s+', " ", text)
        text = text.strip()
        # devolver texto limpio
        return text

    @staticmethod
    def extract_hashtags(text: str) -> list:
        """
        Extrae hashtags de un texto.
        """
        # validar input
        if not isinstance(text, str):
            return []
        # aplicar regex
        hashtags = re.findall(r'#\w+', text)
        # devolver lista
        return hashtags

    @staticmethod
    def extract_keywords(text: str) -> list:
        """
        Extrae palabras clave eliminando stopwords básicas.
        """
        if not isinstance(text, str):
            return []

        stopwords = {
            "the", "and", "is", "in", "to", "of", "for", "on", "with",
            "this", "that", "a", "an", "rt"
        }

        words = text.split()
        keywords = [w for w in words if w not in stopwords and not w.startswith("#")]

        return keywords

    def analytics_hashtags_extended(self) -> dict:
        """
        Realiza análisis global, por usuario y por fecha de los hashtags.
        """

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        df = self.data.copy()
        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df["hashtags"] = df["cleaned_text"].apply(self.extract_hashtags)
        df["date"] = pd.to_datetime(
            df["date"],
            format="%a %b %d %H:%M:%S %z %Y",
            errors="coerce"
        ).dt.date
        df_exploded = df.explode("hashtags").dropna(subset=["hashtags"])

        overall = (
            df_exploded["hashtags"]
            .value_counts()
            .reset_index()
        )
        overall.columns = ["hashtag", "frequency"]

        by_user = (
            df_exploded
            .groupby(["user_name", "hashtags"])
            .size()
            .reset_index(name="frequency")
        )
        by_user.columns = ["user_name", "hashtag", "frequency"]

        by_date = (
            df_exploded
            .groupby(["date", "hashtags"])
            .size()
            .reset_index(name="frequency")
        )
        by_date.columns = ["date", "hashtag", "frequency"]

        df["keywords"] = df["cleaned_text"].apply(self.extract_keywords)

        df_keywords = df.explode("keywords").dropna(subset=["keywords"])

        keywords_overall = (
            df_keywords["keywords"]
            .value_counts()
            .reset_index()
        )

        keywords_overall.columns = ["keyword", "frequency"]

        return {'overall': overall, 'by_user': by_user, 'by_date': by_date, 'keywords_overall': keywords_overall}

    def generate_hashtag_wordcloud(self, overall_df: pd.DataFrame = None, max_words: int = 100, figsize: tuple = (10, 6)) -> None:
        """
        Genera una wordcloud de hashtags.
        """

        if overall_df is None:
            if self.data is None:
                raise ValueError("Data not loaded. Please call load_data() first.")
            overall_df = self.analytics_hashtags_extended()['overall']

        hashtag_freq = dict(zip(overall_df["hashtag"], overall_df["frequency"]))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=max_words
        ).generate_from_frequencies(hashtag_freq)

        plt.figure(figsize=figsize)
        plt.title("WordCloud de Hashtags")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def generate_keyword_wordcloud(self, keywords_df: pd.DataFrame = None, max_words: int = 100):
        if keywords_df is None:
            keywords_df = self.analytics_hashtags_extended()['keywords_overall']

        keyword_freq = dict(zip(keywords_df["keyword"], keywords_df["frequency"]))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=max_words
        ).generate_from_frequencies(keyword_freq)

        plt.figure(figsize=(10, 6))
        plt.title("WordCloud de Keywords")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def save_results(self, output_dir: str = "output"):
        """
        Guarda el dataset limpio y los resultados del análisis en CSV (UTF-8).
        """

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Crear carpeta si no existe
        os.makedirs(output_dir, exist_ok=True)

        # DATASET LIMPIO
        df = self.data.copy()
        df["cleaned_text"] = df["text"].apply(self.clean_text)

        df.to_csv(
            os.path.join(output_dir, "cleaned_dataset.csv"),
            index=False,
            encoding="utf-8"
        )

        # RESULTADOS ANÁLISIS
        results = self.analytics_hashtags_extended()

        results["overall"].to_csv(
            os.path.join(output_dir, "hashtags_overall.csv"),
            index=False,
            encoding="utf-8"
        )

        results["by_user"].to_csv(
            os.path.join(output_dir, "hashtags_by_user.csv"),
            index=False,
            encoding="utf-8"
        )

        results["by_date"].to_csv(
            os.path.join(output_dir, "hashtags_by_date.csv"),
            index=False,
            encoding="utf-8"
        )

    def load_data_api(self, query: str, max_results: int = 100, output_file: str = "data/tweets_from_api.csv") -> pd.DataFrame:
        """
        Conecta con la API de Twitter a través de RapidAPI, extrae tweets
        y guarda una copia local en CSV.
        """

        api_key = os.getenv("RAPIDAPI_KEY")
        api_host = os.getenv("RAPIDAPI_HOST", "twitter-api45.p.rapidapi.com")

        if not api_key:
            raise ValueError(
                "RapidAPI key not found. Please define RAPIDAPI_KEY as an environment variable."
            )

        url = "https://twitter-api45.p.rapidapi.com/search.php"

        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": api_host,
            "Content-Type": "application/json"
        }

        params = {
            "query": query,
            "search_type": "Top"
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error connecting to RapidAPI Twitter endpoint: {e}")

        try:
            json_data = response.json()
        except ValueError:
            raise RuntimeError("The API response is not valid JSON.")

        if json_data.get("status") != "ok":
            raise RuntimeError("The API response status is not OK.")

        tweets = json_data.get("timeline")

        if not tweets or not isinstance(tweets, list):
            raise RuntimeError("The API returned no tweets.")

        rows = []
        for tweet in tweets[:max_results]:
            if tweet.get("type") != "tweet":
                continue

            rows.append({
                "tweet_id": tweet.get("tweet_id"),
                "user_name": tweet.get("screen_name"),
                "date": tweet.get("created_at"),
                "text": tweet.get("text"),
                "lang": tweet.get("lang"),
                "favorites": tweet.get("favorites"),
                "retweets": tweet.get("retweets"),
                "replies": tweet.get("replies"),
                "views": tweet.get("views")
            })

        df = pd.DataFrame(rows)

        if df.empty:
            raise RuntimeError("No valid tweet rows were extracted from the API response.")

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_file, index=False, encoding="utf-8")

        self.data = df
        self.source_file = output_file

        return self.data

    def ensure_clean_text(self):
        """
        Asegura que existe la columna cleaned_text.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")

        if "cleaned_text" not in self.data.columns:
            self.data["cleaned_text"] = self.data["text"].apply(self.clean_text)

    def model_topics(self, num_topics: int = 5, passes: int = 10, num_words: int = 5) -> dict:
        """
        Aplica LDA con gensim para descubrir tópicos en el corpus.
        """

        self.ensure_clean_text()

        import nltk
        from nltk.corpus import stopwords
        from gensim import corpora
        from gensim.models.ldamodel import LdaModel
        from gensim.models import CoherenceModel
        from gensim.utils import simple_preprocess
        import numpy as np
        import re

        nltk.download("stopwords", quiet=True)

        # Stopwords base en inglés, porque los tweets de la API suelen venir en inglés
        stop_words = set(stopwords.words("english"))

        # Stopwords específicas para Twitter y para este dataset
        custom_stopwords = {
            "rt", "http", "https", "co", "amp",
            "you", "your", "yours", "me", "my", "we", "our",
            "he", "his", "him", "she", "her", "they", "them",
            "was", "were", "are", "is", "be", "been",
            "has", "have", "had", "do", "does", "did",
            "will", "would", "can", "could", "should",
            "one", "only", "just", "like", "get", "new"
        }

        stop_words.update(custom_stopwords)

        documents = []

        for text in self.data["cleaned_text"].dropna():
            text = text.lower()

            # Eliminar hashtags como símbolo, pero conservar la palabra
            text = text.replace("#", "")

            # Eliminar caracteres especiales y números
            text = re.sub(r"[\W_]+", " ", text)

            tokens = simple_preprocess(text, deacc=False)

            tokens = [
                word for word in tokens
                if word not in stop_words and len(word) > 2
            ]

            if tokens:
                documents.append(tokens)

        if not documents:
            raise ValueError("No valid documents available for topic modeling.")

        dictionary = corpora.Dictionary(documents)

        # Filtrar términos demasiado raros o demasiado frecuentes
        dictionary.filter_extremes(
            no_below=2,
            no_above=0.6
        )

        corpus_bow = [dictionary.doc2bow(doc) for doc in documents]

        lda_model = LdaModel(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            update_every=1,
            chunksize=100,
            alpha="auto",
            per_word_topics=True
        )

        topics = []

        for idx, topic in lda_model.show_topics(formatted=False, num_words=num_words):
            topic_words = [word for word, _ in topic]
            topics.append(topic_words)
            print(f"Tópico #{idx}: {topic_words}")

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=documents,
            dictionary=dictionary,
            coherence="c_v"
        )

        coherence_score = coherence_model.get_coherence()
        log_perplexity = lda_model.log_perplexity(corpus_bow)
        perplexity = np.exp(-log_perplexity)

        print(f"\nCoherencia: {coherence_score}")
        print(f"Perplexity: {perplexity}")

        return {
            "topics": topics,
            "lda_model": lda_model,
            "corpus_bow": corpus_bow,
            "dictionary": dictionary,
            "documents": documents,
            "coherence": coherence_score,
            "perplexity": perplexity
        }


    def analyze_sentiment(self, model: str = "en_core_web_sm") -> pd.DataFrame:
        """
        Analiza sentimiento usando spaCy + spacytextblob.
        """

        self.ensure_clean_text()

        nlp = spacy.load(model)

        if "spacytextblob" not in nlp.pipe_names:
            nlp.add_pipe("spacytextblob")

        polarities = []
        subjectivities = []

        for text in self.data["cleaned_text"].fillna(""):
            doc = nlp(text)
            polarities.append(doc._.blob.polarity)
            subjectivities.append(doc._.blob.subjectivity)

        self.data["sentiment_polarity"] = polarities
        self.data["sentiment_subjectivity"] = subjectivities

        self.data["sentiment_label"] = self.data["sentiment_polarity"].apply(
            lambda x: "positive" if x > 0.1 else "negative" if x < -0.1 else "neutral"
        )

        return self.data

    def parse_and_summarize(self, summary_ratio: float = 0.3, model: str = "en_core_web_sm") -> str:
        """
        Genera un resumen extractivo del corpus usando spaCy.

        Pasos:
        1. Concatena todos los textos limpios.
        2. Divide el texto en oraciones.
        3. Calcula frecuencia de palabras relevantes.
        4. Puntúa cada oración según esas frecuencias.
        5. Selecciona las oraciones más representativas.
        """

        self.ensure_clean_text()

        nlp = spacy.load(model)

        corpus_text = " ".join(self.data["cleaned_text"].dropna().tolist())
        doc = nlp(corpus_text)

        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return ""

        words = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]

        word_freq = Counter(words)

        if not word_freq:
            return ""

        max_freq = max(word_freq.values())

        sentence_scores = {}

        for i, sentence in enumerate(sentences):
            sent_doc = nlp(sentence)
            score = 0

            for token in sent_doc:
                lemma = token.lemma_.lower()
                if lemma in word_freq:
                    score += word_freq[lemma] / max_freq

            sentence_scores[i] = score

        top_n = max(1, math.ceil(len(sentences) * summary_ratio))

        selected_indexes = sorted(
            [
                idx for idx, _ in sorted(
                sentence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            ]
        )

        summary = " ".join(sentences[i] for i in selected_indexes)

        return summary

    def parse_documents(self, model: str = "en_core_web_sm") -> pd.DataFrame:
        """
        Aplica parsing con spaCy sobre los textos limpios.

        Extrae:
        - información de tokens
        - entidades nombradas
        - sujeto, verbo y objeto
        - conteo de categorías gramaticales
        - frecuencia de verbos
        """

        self.ensure_clean_text()

        nlp = spacy.load(model)

        def parse_and_extract(text):
            doc = nlp(text)

            tokens_info = []
            for token in doc:
                tokens_info.append({
                    "text": token.text,
                    "dep": token.dep_,
                    "pos": token.pos_,
                    "head": token.head.text
                })

            entities_info = [(ent.text, ent.label_) for ent in doc.ents]

            return {
                "text": text,
                "tokens": tokens_info,
                "entities": entities_info
            }

        def find_svo(text):
            doc = nlp(text)

            subject = None
            verb = None
            object_ = None

            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    verb = token.text
                elif token.dep_ in ["nsubj", "nsubjpass"]:
                    subject = token.text
                elif token.dep_ in ["obj", "iobj", "dobj", "pobj", "attr"]:
                    object_ = token.text

            return subject, verb, object_

        def count_pos(text):
            doc = nlp(text)
            return Counter([token.pos_ for token in doc])

        def verb_frequency(text):
            doc = nlp(text)
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            return dict(Counter(verbs))

        df = self.data.copy()

        df["parsing_info"] = df["cleaned_text"].apply(parse_and_extract)

        df[["subject", "verb", "object"]] = df["cleaned_text"].apply(
            lambda x: pd.Series(find_svo(x))
        )

        df["pos_counts"] = df["cleaned_text"].apply(count_pos)

        df["num_nouns"] = df["pos_counts"].apply(
            lambda x: x.get("NOUN", 0) + x.get("PROPN", 0)
        )
        df["num_verbs"] = df["pos_counts"].apply(lambda x: x.get("VERB", 0))
        df["num_adjectives"] = df["pos_counts"].apply(lambda x: x.get("ADJ", 0))
        df["num_adverbs"] = df["pos_counts"].apply(lambda x: x.get("ADV", 0))

        df["verb_frequency"] = df["cleaned_text"].apply(verb_frequency)

        self.data = df

        return self.data

    def export_all_results(self, output_dir="output"):
        """
        Exporta todos los resultados del análisis a CSV/TXT.
        """

        import os
        os.makedirs(output_dir, exist_ok=True)

        # -----------------------
        # DATASET LIMPIO
        # -----------------------
        df = self.data.copy()
        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df.to_csv(f"{output_dir}/cleaned_dataset.csv", index=False, encoding="utf-8")

        # -----------------------
        # HASHTAGS + KEYWORDS
        # -----------------------
        results = self.analytics_hashtags_extended()

        results["overall"].to_csv(f"{output_dir}/hashtags_overall.csv", index=False)
        results["by_user"].to_csv(f"{output_dir}/hashtags_by_user.csv", index=False)
        results["by_date"].to_csv(f"{output_dir}/hashtags_by_date.csv", index=False)
        results["keywords_overall"].to_csv(f"{output_dir}/keywords_overall.csv", index=False)

        # -----------------------
        # SENTIMIENTO
        # -----------------------
        df_sentiment = self.analyze_sentiment()
        df_sentiment.to_csv(f"{output_dir}/sentiment_analysis.csv", index=False)

        # -----------------------
        # TOPICS
        # -----------------------
        lda_results = self.model_topics(num_topics=4, passes=10)

        topics_df = pd.DataFrame({
            "topic": [f"Tópico {i}" for i in range(len(lda_results["topics"]))],
            "keywords": [", ".join(topic) for topic in lda_results["topics"]]
        })

        topics_df.to_csv(f"{output_dir}/topics.csv", index=False)

        # -----------------------
        # PARSING
        # -----------------------
        df_parsing = self.parse_documents()
        df_parsing.to_csv(f"{output_dir}/parsing_results.csv", index=False)

        # -----------------------
        # RESUMEN
        # -----------------------
        summary = self.parse_and_summarize(summary_ratio=0.2)

        with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)