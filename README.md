# 📊 Análisis de Sentimiento y Tendencias en Redes Sociales (U4)

Proyecto desarrollado como parte de la asignatura **Búsqueda y Análisis de la Información**.

En esta fase se amplía el pipeline desarrollado en la Unidad 2, incorporando:
- integración con APIs externas (Twitter vía RapidAPI)
- técnicas avanzadas de análisis de texto (NLP)

---

## 🚀 Fase actual: U4 - APIs y Análisis Avanzado

En esta entrega se extiende la clase `DataExtractor` para trabajar con datos más reales desde una API externa y aplicar técnicas avanzadas de análisis textual.

---

# 🔌 1. Integración con API de Twitter de RapidAPI

## 🧩 Descripción

Se ha implementado el método `load_data_api()`, que permite conectarse a la API de Twitter a través de RapidAPI para extraer tweets a partir de una consulta específica.

Este método amplía el pipeline original permitiendo trabajar no solo con datasets estáticos, sino también con datos dinámicos.

---

## ⚙️ Funcionamiento

El método realiza los siguientes pasos:

1. **Conexión a la API**
   - Se utiliza la librería `requests` para realizar una petición HTTP GET
   - La API utilizada es `twitter-api45` a través de RapidAPI

2. **Parámetros de búsqueda**
   - `query`: palabra clave o tema a buscar
   - `max_results`: número máximo de tweets a recuperar

3. **Autenticación segura**
   - La clave de la API se gestiona mediante variables de entorno (`RAPIDAPI_KEY`)
   - No se incluye información sensible en el código

4. **Validación de la respuesta**
   - Se comprueba el código de estado HTTP
   - Se valida que la respuesta JSON tenga estado `"ok"`
   - Se verifica la existencia de datos en la respuesta

5. **Transformación de datos**
   - Se extraen los campos relevantes de cada tweet:
     - `tweet_id`
     - `user_name`
     - `date`
     - `text`
     - métricas (likes, retweets, etc.)
   - Se estructura la información en un `DataFrame`

6. **Filtrado temporal**
   - Conversión de la fecha a formato datetime
   - Aplicación opcional de filtros por rango de fechas

7. **Almacenamiento**
   - Los datos se almacenan en:
     - `self.data` para uso en el pipeline
     - un archivo CSV en UTF-8 para reutilización posterior

---

## 🛠 Uso del método

```python
extractor = DataExtractor()

df_api = extractor.load_data_api(
    query="bitcoin",
    max_results=20,
    output_file="data/tweets_from_api.csv"
)

df_api.head()
```

---

---

# 🧠 2. Modelado y Análisis Avanzado

En esta fase se aplican distintas técnicas de Procesamiento de Lenguaje Natural (NLP) sobre los tweets obtenidos.

Estas técnicas permiten transformar texto no estructurado en información analizable.

---

## 🧩 2.1 Modelado de tópicos (LDA)

### 🔍 Descripción

Se ha aplicado **Latent Dirichlet Allocation (LDA)** utilizando la librería `gensim` para descubrir los principales temas presentes en los tweets.

### ⚙️ Metodología

- Tokenización de textos (`simple_preprocess`)
- Eliminación de stopwords
- Creación de:
  - diccionario (`Dictionary`)
  - corpus en formato Bag of Words
- Entrenamiento del modelo LDA:
  - número de tópicos configurable
  - número de iteraciones (`passes`)

### 📊 Resultados

Se obtienen:

- palabras clave por tópico
- métricas de evaluación:
  - **Coherencia**
  - **Perplexity**

Además, se utiliza `pyLDAvis` para visualizar:

- distancia entre tópicos
- términos más relevantes

### 🧠 Interpretación

Los tópicos identificados reflejan:

- comportamiento del mercado cripto
- comparativas con activos tradicionales
- noticias relevantes del sector
- dinámica de precios de Bitcoin

---

## 😊 2.2 Análisis de sentimiento

### 🔍 Descripción

Se realiza análisis de sentimiento utilizando **spaCy + spacytextblob**.

### ⚙️ Metodología

- Procesamiento de cada tweet con spaCy
- Obtención de:
  - **polaridad** (-1 a 1)
  - **subjetividad** (0 a 1)
- Clasificación en:
  - positivo
  - negativo
  - neutro

### 📊 Resultados

Se obtiene:

- dataset enriquecido con métricas de sentimiento
- distribución de sentimientos mediante gráfico

### 🧠 Interpretación

Los resultados muestran que:

- predomina el sentimiento **neutral/positivo**
- los tweets suelen ser informativos o analíticos
- hay baja presencia de opiniones extremadamente negativas

---

## 🔍 2.3 Parsing sintáctico

### 🔍 Descripción

Se utiliza **spaCy** para analizar la estructura gramatical de los textos.

### ⚙️ Técnicas aplicadas

- extracción de relaciones **Sujeto-Verbo-Objeto (SVO)**
- conteo de categorías gramaticales (POS):
  - sustantivos
  - verbos
  - adjetivos
  - adverbios
- frecuencia de verbos

### 📊 Resultados

Se genera un dataset con:

- sujeto, verbo y objeto por frase
- estadísticas lingüísticas
- frecuencia de acciones (verbos)

Además, se visualizan árboles sintácticos con `displacy`.

### 🧠 Interpretación

Este análisis permite:

- entender cómo se estructuran los mensajes
- identificar acciones relevantes (ej: “buy”, “sell”, “rise”)
- analizar patrones lingüísticos en redes sociales

---

## 📝 2.4 Resumen extractivo

### 🔍 Descripción

Se implementa un método de resumen extractivo que selecciona las frases más relevantes del corpus.

### ⚙️ Metodología

- tokenización de frases
- cálculo de frecuencia de palabras
- puntuación de frases en función de su relevancia
- selección de las frases más representativas

### 📊 Resultados

Se obtiene un resumen que condensa la información clave de los tweets.

### 🧠 Interpretación

El resumen permite:

- entender rápidamente el contenido del dataset
- identificar temas principales sin leer todos los tweets
- mejorar la eficiencia del análisis

---

# 📊 3. Resultados del análisis

## 🔹 Hashtags

El análisis de hashtags no produce resultados en este dataset.

Esto se debe a que los tweets obtenidos desde la API no contienen hashtags explícitos en el texto ni en los metadatos.

No se trata de un error del pipeline, sino de una característica del dataset.

---

## 🔹 Keywords

Se utilizan como alternativa a los hashtags.

Permiten identificar términos frecuentes como:

- bitcoin
- crypto
- market
- price

---

## 🔹 Tópicos

Los tópicos obtenidos reflejan:

- ciclos de mercado
- activos financieros alternativos
- eventos relevantes del ecosistema cripto

---

## 🔹 Sentimiento

Distribución:

- mayoritariamente neutral
- presencia moderada de sentimiento positivo

---

## 🔹 Parsing

Permite identificar:

- estructura de los mensajes
- acciones más frecuentes
- patrones lingüísticos

---

## 🔹 Resumen

Condensa el contenido del dataset en unas pocas frases clave.

---

## 📦 Exportación de resultados

El proyecto genera automáticamente archivos en la carpeta `output/` que contienen todos los resultados del análisis:

- dataset limpio
- análisis de hashtags y keywords
- análisis de sentimiento
- modelado de tópicos
- parsing sintáctico
- resumen del corpus

Esto permite reproducir y analizar los resultados sin necesidad de ejecutar el código.

# ▶️ 4. Reproducibilidad

## ⚙️ Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## 🔐 Variables de entorno
Configurar la API key:

Windows PowerShell:
```powershell
setx RAPIDAPI_KEY "tu_clave_aqui"
setx RAPIDAPI_HOST "twitter-api45.p.rapidapi.com"
```
## ▶️ Ejecución
1. Ejecutar el notebook:
```bash
jupyter notebook
```
2. Ejecutar las celdas en orden para reproducir el análisis.
3. (Opcional) Ejecutar Streamlit:
```bash
streamlit run app-streamlit.py
```

## 🧠 Conclusiones
En esta segunda fase se ha ampliado significativamente el pipeline inicial incorporando:
- Integración con API externa
- Técnicas avanzadas de NLP
- Análisis estructural del lenguaje

Aunque el dataset de la API presenta limitaciones (como la ausencia de hashtags), se han aplicado técnicas alternativas que permiten extraer información relevante.

El proyecto demuestra la capacidad de transformar datos textuales no estructurados en conocimiento útil mediante técnicas modernas de análisis de datos.

## 👨‍💻 Autor

Víctor Sánchez Grande