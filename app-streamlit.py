import streamlit as st
from data_extractor import DataExtractor

# Cargar datos
extractor = DataExtractor("data/Bitcoin_tweets_dataset_2.csv")
extractor.load_data()

results = extractor.analytics_hashtags_extended()

# Título
st.title("📊 Hashtag Analytics")

# Slider para controlar tamaño
top_n = st.slider("Número de filas a mostrar", 5, 500, 10)

# -----------------------
# OVERALL
# -----------------------
st.subheader("🌍 Top Hashtags (Global)")
st.dataframe(results["overall"].head(top_n))

# -----------------------
# POR USUARIO
# -----------------------
st.subheader("👤 Hashtags por usuario")
st.dataframe(results["by_user"].head(top_n))

# -----------------------
# POR FECHA
# -----------------------
st.subheader("📅 Hashtags por fecha")
st.dataframe(results["by_date"].head(top_n))

# -----------------------
# KEYWORDS
# -----------------------

st.subheader("🔑 Keywords más frecuentes")
st.dataframe(results["keywords_overall"].head(top_n))