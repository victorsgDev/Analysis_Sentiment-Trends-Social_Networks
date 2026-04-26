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
