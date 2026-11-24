# 🤖 Procesamiento de Datasets en Machine Learning

Aplicación web interactiva desarrollada con **Streamlit** para demostrar las etapas fundamentales del preprocesamiento de datos en Machine Learning utilizando **Pandas**, **Scikit-learn**, **Matplotlib** y **Seaborn**.

## 📋 Características

La aplicación cuenta con un menú lateral que permite navegar entre tres casos de estudio clásicos:

1. **🚢 Ejercicio 1: Dataset Titanic**
   * **Objetivo:** Preparar los datos para predecir la supervivencia de los pasajeros.
   * **Procesos:** Carga de archivo CSV (`titanic.csv`), exploración inicial, eliminación de columnas irrelevantes (`Name`, `Ticket`, `Cabin`), imputación de valores nulos (media para `Age` y moda para `Embarked`), codificación de variables (`LabelEncoder` y `One-Hot Encoding`), estandarización con `StandardScaler` y división Train/Test (70/30).

2. **🎓 Ejercicio 2: Dataset Student Performance**
   * **Objetivo:** Procesar datos académicos para predecir la calificación final (`G3`).
   * **Procesos:** Carga de CSV separado por punto y coma (`student-mat.csv`), eliminación de duplicados, codificación `One-Hot` para variables categóricas, normalización con `MinMaxScaler`, división Train/Test (80/20) y visualización de matriz de correlación entre notas (`G1`, `G2`, `G3`).

3. **🌸 Ejercicio 3: Dataset Iris**
   * **Objetivo:** Implementar un flujo completo de preprocesamiento y visualización con datos integrados.
   * **Procesos:** Carga directa desde `sklearn.datasets`, estandarización con `StandardScaler`, división Train/Test (70/30), gráfico de dispersión (*Sepal Length* vs *Petal Length*) y estadísticas descriptivas posteriores al escalado.

## 🛠️ Tecnologías Utilizadas

* [Python 3.8+](https://python.org)
* [Streamlit](https://streamlit.io) - Interfaz web interactiva
* [Pandas](https://pydata.org) & [NumPy](https://numpy.org) - Manipulación y análisis de datos
* [Scikit-learn](https://scikit-learn.org) - Preprocesamiento y partición de datos
* [Matplotlib](https://matplotlib.org) & [Seaborn](https://pydata.org) - Visualización de datos

## 🚀 Instalación y Ejecución

1. **Clona este repositorio:**
   ```bash
   git clone https://github.com
   cd tu-repositorio
   ```

2. **Crea un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instala las dependencias:**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```

4. **Ejecuta la aplicación:**
   ```bash
   streamlit run app.py
   ```
   *(Nota: reemplaza `app.py` por el nombre real de tu archivo Python).*

## 📂 Archivos de Datos Necesarios

Para probar los dos primeros módulos, necesitarás descargar los siguientes datasets públicos:
* `titanic.csv` (Disponible en [Kaggle - Titanic](https://kaggle.com))
* `student-mat.csv` (Disponible en [UCI Machine Learning Repository](https://uci.edu))

El **Ejercicio 3 (Iris)** no requiere archivos externos, ya que se carga automáticamente desde Scikit-learn.
