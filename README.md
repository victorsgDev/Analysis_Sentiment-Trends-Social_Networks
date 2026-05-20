# 🕸️ U6 - Análisis de Redes Sociales e Integración con LLM

En esta fase se amplía el pipeline desarrollado en las unidades anteriores incorporando técnicas de **análisis de redes sociales** mediante `NetworkX` y la integración con un **modelo LLM local** para generar una interpretación automática de los resultados obtenidos.

La práctica parte de los tweets extraídos mediante la API de Twitter/RapidAPI y reutiliza el dataset generado en fases anteriores para construir una red de interacciones entre usuarios.

---

## 🧩 Objetivos de esta fase

Los principales objetivos de esta entrega son:

- Construir un grafo de interacciones entre usuarios a partir de menciones en tweets.
- Calcular métricas de red relevantes mediante `NetworkX`.
- Detectar comunidades dentro del grafo.
- Extraer información clave de la red, como usuarios más centrales y términos frecuentes.
- Generar un prompt estructurado con los resultados del análisis.
- Ejecutar un modelo LLM local para obtener una interpretación en lenguaje natural del comportamiento de la red.
- Integrar los resultados en el pipeline existente y en la visualización con Streamlit.

---

## 🔌 1. Dataset utilizado

El análisis se realiza sobre los tweets obtenidos mediante el método `load_data_api()`.

Este método permite recuperar datos desde la API `twitter-api45` de RapidAPI, utilizando paginación mediante el parámetro `cursor`.

La API limita el número de tweets devueltos por petición, por lo que se ha implementado un sistema de múltiples llamadas usando el valor `next_cursor` devuelto por la respuesta.

Ejemplo de uso:

~~~python
extractor = DataExtractor()

df_api = extractor.load_data_api(
    query="bitcoin",
    api_calls=10,
    output_file="data/tweets_from_api.csv"
)

df_api.head()
~~~

De esta forma, en lugar de depender únicamente de una respuesta limitada a unos pocos tweets, el pipeline puede recuperar varias páginas de resultados y construir un dataset más representativo.

---

## 🕸️ 2. Análisis de Redes Sociales con NetworkX

### Descripción

Se ha implementado un análisis de red a partir de las menciones presentes en los tweets.

Cada nodo del grafo representa un usuario y cada arista representa una interacción de mención:

~~~text
autor del tweet  --->  usuario mencionado
~~~

Por ejemplo, si el usuario `A` escribe un tweet mencionando a `@B`, se genera una arista dirigida desde `A` hacia `B`.

---

### Metodología

El proceso seguido es:

1. Extraer menciones de cada tweet usando expresiones regulares.
2. Construir un grafo dirigido (`DiGraph`) con `NetworkX`.
3. Añadir nodos para los autores de los tweets.
4. Añadir aristas hacia los usuarios mencionados.
5. Incrementar el peso de la arista si una interacción se repite.
6. Calcular métricas de red.
7. Detectar comunidades.
8. Visualizar el grafo completo y las principales comunidades.

---

### Métodos implementados

Se han añadido métodos específicos a la clase `DataExtractor`:

- `extract_mentions()`: extrae menciones de usuario a partir del texto.
- `build_interaction_graph()`: construye el grafo dirigido de interacciones.
- `analyze_network()`: calcula métricas de red y comunidades.
- `visualize_network()`: genera una visualización del grafo completo.
- `generate_network_insights_prompt()`: genera el prompt para el LLM.
- `analyze_network_with_llm()`: ejecuta el análisis interpretativo con un modelo LLM local.

---

## 📊 3. Métricas de red calculadas

El método `analyze_network()` calcula diferentes métricas para estudiar el papel de cada usuario dentro de la red.

Las métricas principales son:

- **degree**: número total de conexiones de un nodo.
- **in_degree**: número de menciones recibidas.
- **out_degree**: número de menciones realizadas.
- **degree_centrality**: centralidad de grado.
- **in_degree_centrality**: centralidad basada en menciones recibidas.
- **out_degree_centrality**: centralidad basada en menciones realizadas.
- **betweenness_centrality**: mide si un nodo actúa como puente entre distintas partes de la red.

Ejemplo de ejecución:

~~~python
network_results = extractor.analyze_network()

network_results["metrics"].head(10)
~~~

---

## 👥 4. Detección de comunidades

Para detectar comunidades se utilizan los componentes débilmente conectados del grafo.

Esto permite identificar grupos de usuarios conectados entre sí por menciones, aunque la dirección de la relación no sea estrictamente considerada para formar la comunidad.

Ejemplo:

~~~python
communities = network_results["communities"]

top_communities = sorted(
    communities,
    key=len,
    reverse=True
)[:5]

for i, community in enumerate(top_communities):
    print(f"Comunidad {i+1}:")
    print(list(community)[:10])
    print()
~~~

---

## 🧭 5. Visualización del grafo y comunidades

Además del grafo completo, se visualizan las comunidades principales de forma independiente.

Esto permite interpretar mejor la estructura del grafo, ya que una red completa puede ser difícil de leer cuando contiene muchos nodos.

Ejemplo de visualización por comunidades:

~~~python
import matplotlib.pyplot as plt
import networkx as nx

graph = network_results["graph"]
communities = network_results["communities"]

top_communities = sorted(
    communities,
    key=len,
    reverse=True
)[:5]

for i, community in enumerate(top_communities, start=1):
    subgraph = graph.subgraph(community)

    plt.figure(figsize=(10, 6))

    pos = nx.spring_layout(subgraph, seed=42)

    node_sizes = [
        max(500 * subgraph.degree(node), 500)
        for node in subgraph.nodes()
    ]

    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_size=node_sizes,
        alpha=0.7
    )

    nx.draw_networkx_edges(
        subgraph,
        pos,
        arrows=True,
        alpha=0.4
    )

    nx.draw_networkx_labels(
        subgraph,
        pos,
        font_size=9
    )

    plt.title(
        f"Comunidad {i} - {subgraph.number_of_nodes()} nodos / {subgraph.number_of_edges()} aristas"
    )

    plt.axis("off")
    plt.show()
~~~

---

## 🏆 6. Información clave extraída

A partir del análisis de red se obtiene información relevante como:

- número total de nodos,
- número total de aristas,
- densidad de la red,
- número de comunidades detectadas,
- top 3 usuarios con mayor centralidad,
- hashtag más frecuente,
- keywords más frecuentes.

Ejemplo:

~~~python
network_results["summary"]
~~~

En caso de que el dataset no contenga hashtags explícitos, el sistema lo refleja correctamente y utiliza las keywords como alternativa para analizar los términos más relevantes del corpus.

---

## 🤖 7. Generación de Prompt y Análisis con LLM Local

### Descripción

La información obtenida del análisis de red se integra en un prompt estructurado que posteriormente se pasa a un modelo LLM local.

El modelo utilizado es el indicado en el enunciado de la práctica:

~~~text
google/gemma-4-E2B-it
~~~

El objetivo del LLM es generar una interpretación en lenguaje natural sobre el comportamiento de la red.

---

### Información incluida en el prompt

El prompt incorpora:

- número de nodos,
- número de aristas,
- densidad de la red,
- número de comunidades,
- top 3 usuarios más centrales,
- comunidades principales,
- hashtag más frecuente,
- keywords más frecuentes.

Ejemplo de generación del prompt:

~~~python
prompt = extractor.generate_network_insights_prompt(network_results)

print(prompt)
~~~

---

### Ejecución del modelo LLM local

El análisis interpretativo se genera mediante `transformers`:

~~~python
llm_analysis = extractor.analyze_network_with_llm(
    network_results=network_results,
    model_name="google/gemma-4-E2B-it",
    max_new_tokens=500
)

print(llm_analysis)
~~~

---

### Interpretación generada

El modelo LLM recibe los resultados estructurados del análisis de red y genera una explicación sobre:

- el nivel de fragmentación de la red,
- la importancia de los usuarios centrales,
- la distribución de comunidades,
- la relación entre términos frecuentes y comportamiento social,
- conclusiones generales sobre la dinámica observada.

En los resultados obtenidos, el modelo destaca que la red presenta una estructura muy fragmentada, formada por muchas comunidades pequeñas, lo que sugiere que la conversación no se organiza alrededor de un único grupo central, sino en múltiples nichos de interacción.

---

## 📦 8. Exportación de resultados

Para mejorar la reutilización de los resultados, se mantiene la exportación de datos procesados y análisis generados.

Se recomienda almacenar en la carpeta `output/` archivos como:

~~~text
output/
├── cleaned_dataset.csv
├── hashtags_overall.csv
├── keywords_overall.csv
├── sentiment_analysis.csv
├── topics.csv
├── parsing_results.csv
├── network_metrics.csv
├── network_summary.csv
├── network_top_3_users.csv
├── llm_network_analysis.txt
└── summary.txt
~~~

Esto permite revisar los resultados sin necesidad de ejecutar todo el pipeline desde cero.

---

## 🖥️ 9. Integración con Streamlit

La aplicación Streamlit se amplía para incluir resultados de la Unidad 6.

Además de las funcionalidades previas de hashtags, keywords, sentimiento, tópicos y resumen, se añaden:

- análisis de red,
- métricas generales del grafo,
- top usuarios por centralidad,
- comunidades principales,
- generación de prompt para LLM,
- ejecución opcional del análisis interpretativo con LLM local.

Ejemplo de ejecución:

~~~bash
streamlit run app-streamlit.py
~~~

---

## ▶️ 10. Reproducibilidad

### Instalación de dependencias

Para instalar las dependencias necesarias:

~~~bash
pip install -r requirements.txt
~~~

Además, para ejecutar el modelo LLM local, se necesitan las librerías de Hugging Face:

~~~bash
pip install transformers accelerate torch
~~~

---

### Variables de entorno

La API key de RapidAPI debe gestionarse mediante variables de entorno y no debe incluirse en el código.

En Windows PowerShell:

~~~powershell
$env:RAPIDAPI_KEY="tu_api_key"
$env:RAPIDAPI_HOST="twitter-api45.p.rapidapi.com"
~~~

Si se desea usar Hugging Face autenticado:

~~~powershell
$env:HF_TOKEN="tu_huggingface_token"
~~~

---

### Ejecución recomendada

1. Crear o activar el entorno virtual.
2. Instalar dependencias con `requirements.txt`.
3. Configurar las variables de entorno necesarias.
4. Ejecutar el notebook principal.
5. Revisar los resultados generados.
6. Ejecutar opcionalmente la app de Streamlit.

~~~bash
jupyter notebook
~~~

~~~bash
streamlit run app-streamlit.py
~~~

---

## 🧠 11. Conclusión

En esta tercera entrega se amplía el pipeline anterior incorporando análisis de redes sociales e integración con LLMs.

El análisis con `NetworkX` permite transformar menciones entre usuarios en una estructura de grafo interpretable, calcular métricas de influencia y detectar comunidades.

La integración con un modelo LLM local permite convertir los resultados numéricos del análisis de red en una explicación en lenguaje natural, facilitando la interpretación del comportamiento social observado.

Con esta fase, el proyecto evoluciona desde un pipeline de extracción y análisis textual hacia un sistema más completo que combina:

- extracción de datos desde API,
- procesamiento NLP,
- análisis de redes,
- visualización,
- e interpretación automática mediante LLMs.