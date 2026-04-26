# 📊 Análisis de Sentimiento y Tendencias en Redes Sociales

Proyecto desarrollado como parte de la asignatura **Busqueda y Análisis de la Información**.

Este proyecto tiene como objetivo construir un pipeline de análisis de datos textuales procedentes de redes sociales, centrándose en la extracción, limpieza y análisis de información relevante como hashtags.

---

## 🚀 Fase actual: U2 - Extracción y Tratamiento de Datos

En esta primera entrega se ha implementado la fase inicial del proyecto, que incluye:

- Extracción de datos desde un dataset de tweets
- Limpieza y normalización del texto
- Extracción de hashtags
- Análisis de frecuencia de hashtags
- Visualización mediante WordCloud

---

## 📁 Dataset

Se ha utilizado el dataset:

- **Bitcoin Tweets Dataset**
- Fuente: https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets/data?select=Bitcoin_tweets_dataset_2.csv

El dataset contiene tweets relacionados con Bitcoin, incluyendo información como:
- texto del tweet
- usuario
- fecha

---

## ⚙️ Funcionalidades implementadas

Se ha desarrollado la clase `DataExtractor`, que encapsula todo el pipeline:

### 🔹 Carga de datos
- Lectura de archivos CSV
- Manejo de datasets grandes

### 🔹 Exportación de resultados

El proyecto genera automáticamente archivos CSV en la carpeta `output/`:

- `cleaned_dataset.csv`: dataset limpio
- `hashtags_overall.csv`: frecuencia global de hashtags
- `hashtags_by_user.csv`: frecuencia por usuario
- `hashtags_by_date.csv`: evolución temporal

Esto permite analizar los resultados sin necesidad de ejecutar el código.

### 🔹 Limpieza de texto
Se aplican expresiones regulares para:
- eliminar URLs
- eliminar menciones (@user)
- eliminar caracteres especiales
- normalizar a minúsculas
- eliminar espacios redundantes
- conservar hashtags

### 🔹 Extracción de hashtags
- Uso de regex (`#\w+`)
- Conversión de texto a estructura de datos (listas)

### 🔹 Análisis de datos
Se realiza análisis de frecuencia en tres niveles:

- **Global (`overall`)**
- **Por usuario (`by_user`)**
- **Por fecha (`by_date`)**

Se utilizan operaciones de pandas como:
- `apply`
- `explode`
- `groupby`

### 🔹 Visualización
- Generación de WordCloud de hashtags más frecuentes
- Visualización interactiva mediante Streamlit

---

## 🛠 Tecnologías utilizadas

- Python 3
- Pandas
- Regex (re)
- Matplotlib
- WordCloud
- Streamlit

---

## ▶️ Cómo ejecutar

1. Clonar el repositorio:
```bash
git clone <repo-url>
```

2.1 Instalar dependencias:
```bash
pip install pandas matplotlib wordcloud streamlit
```
2.2 Reproducibilidad:
Para facilitar la ejecución del proyecto en otros entornos, se incluye un archivo `requirements.txt` con todas las dependencias necesarias.
```bash
pip install -r requirements.txt
```

3. Ejecutar el notebook (opcional):
- Abrir el archivo `.ipynb`
- Ejecutar las celdas en orden

4. Ejecutar el dashboard con Streamlit:
```bash
streamlit run app-streamlit.py
```

---

## 📊 Resultados

Se obtiene:

- ranking de hashtags más utilizados
- análisis de uso por usuario
- evolución temporal de hashtags
- visualización mediante WordCloud y dashboard interactivo

---

## 🧠 Conclusiones

Durante el análisis se observa que:

- ciertos hashtags aparecen con alta frecuencia
- algunos usuarios concentran gran parte de las menciones (esto podría indicar actividad automatizada (bots))

---

## 🔜 Próximos pasos

En futuras entregas se ampliará el proyecto con:

- análisis de sentimiento
- modelos de NLP
- clasificación de texto
- visualización avanzada (dashboard)

---

## 👨‍💻 Autor

Víctor Sánchez Grande

