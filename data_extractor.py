import pandas as pd
import re
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class DataExtractor:
    def __init__(self, source_file: str, chunksize: int = 100000):
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

    def analytics_hashtags_extended(self) -> dict:
        """
        Realiza análisis global, por usuario y por fecha de los hashtags.
        """

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        df = self.data.copy()
        df["cleaned_text"] = df["text"].apply(self.clean_text)
        df["hashtags"] = df["cleaned_text"].apply(self.extract_hashtags)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df_exploded = df.explode("hashtags").dropna(subset=["hashtags"])

        overall = (
            df_exploded["hashtags"]
            .value_counts()
            .reset_index()
        )
        overall.columns = ["hashtag", "frecuency"]

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

        return {'overall': overall, 'by_user': by_user, 'by_date': by_date}

    def generate_hashtag_wordcloud(self, overall_df: pd.DataFrame = None, max_words: int = 100, figsize: tuple = (10, 6)) -> None:
        """
        Genera una wordcloud de hashtags.
        """

        if overall_df is None:
            if self.data is None:
                raise ValueError("Data not loaded. Please call load_data() first.")
            overall_df = self.analytics_hashtags_extended()['overall']

        hashtag_freq = dict(zip(overall_df["hashtag"], overall_df["frecuency"]))

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