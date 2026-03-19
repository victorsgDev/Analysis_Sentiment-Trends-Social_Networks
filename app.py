import streamlit as st
import pandas as pd

# Simulamos datos (para empezar fácil)
data = {
    "hashtag": ["#bitcoin", "#btc", "#crypto"],
    "frequency": [100, 80, 50]
}

df = pd.DataFrame(data)

# Título
st.title("📊 Hashtag Analytics")

# Texto
st.write("Top hashtags:")

# Mostrar tabla
st.dataframe(df)