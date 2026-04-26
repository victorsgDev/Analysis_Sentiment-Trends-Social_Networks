import streamlit as st
import pandas as pd
from data_extractor import DataExtractor

st.title("📊 Social Media NLP Analytics")

# -----------------------
# CONFIGURACIÓN
# -----------------------
data_file = st.selectbox(
    "Selecciona fuente de datos",
    [
        "data/tweets_from_api.csv",
        "data/Bitcoin_tweets_dataset_2.csv"
    ]
)

top_n = st.slider("Número de filas a mostrar", 5, 100, 10)

# -----------------------
# CARGA DE DATOS
# -----------------------
extractor = DataExtractor(data_file)
extractor.load_data()

st.subheader("📁 Dataset cargado")
st.write(f"Registros cargados: {len(extractor.data)}")
st.dataframe(extractor.data.head(top_n))

# -----------------------
# HASHTAGS Y KEYWORDS
# -----------------------
results = extractor.analytics_hashtags_extended()

st.subheader("🌍 Top Hashtags")
if results["overall"].empty:
    st.info("No se han encontrado hashtags explícitos en este dataset.")
else:
    st.dataframe(results["overall"].head(top_n))

st.subheader("👤 Hashtags por usuario")
if results["by_user"].empty:
    st.info("No se han encontrado hashtags explícitos en este dataset.")
else:
    st.dataframe(results["by_user"].head(top_n))

st.subheader("📅 Hashtags por fecha")
if results["by_date"].empty:
    st.info("No se han encontrado hashtags explícitos en este dataset.")
else:
    st.dataframe(results["by_date"].head(top_n))

st.subheader("🔑 Keywords más frecuentes")
if results["keywords_overall"].empty:
    st.info("No se han encontrado keywords explícitas en este dataset.")
else:
    st.dataframe(results["keywords_overall"].head(top_n))

is_api_data = "tweets_from_api.csv" in data_file

# -----------------------
# SENTIMIENTO
# -----------------------
st.subheader("😊 Análisis de sentimiento")

if not is_api_data:
    st.info("Disponible solo para datos de la API (U4)")
else:
    if st.button("Ejecutar análisis de sentimiento"):
        df_sentiment = extractor.analyze_sentiment()

        st.dataframe(
            df_sentiment[[
                "text",
                "sentiment_polarity",
                "sentiment_subjectivity",
                "sentiment_label"
            ]].head(top_n)
        )

        st.bar_chart(df_sentiment["sentiment_label"].value_counts())

# -----------------------
# TOPIC MODELING
# -----------------------
st.subheader("🧠 Modelado de tópicos (LDA)")

if not is_api_data:
    st.info("Disponible solo para datos de la API (U4)")
else:
    num_topics = st.slider("Número de tópicos", 2, 8, 4)

    if st.button("Ejecutar LDA"):
        lda_results = extractor.model_topics(num_topics=num_topics, passes=10)

        topics_df = pd.DataFrame({
            "topic": [f"Tópico {i}" for i in range(len(lda_results["topics"]))],
            "keywords": [", ".join(topic) for topic in lda_results["topics"]]
        })

        st.dataframe(topics_df)
        st.write(f"Coherencia: {lda_results['coherence']:.4f}")

# -----------------------
# RESUMEN
# -----------------------
st.subheader("📝 Resumen extractivo")

if not is_api_data:
    st.info("Disponible solo para datos de la API (U4)")
else:
    summary_ratio = st.slider("Ratio de resumen", 0.1, 0.5, 0.2)

    if st.button("Generar resumen"):
        summary = extractor.parse_and_summarize(summary_ratio=summary_ratio)
        st.write(summary)