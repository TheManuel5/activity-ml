import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Procesamiento de Datasets - ML", layout="wide")

st.title("🤖 Procesamiento de Datasets en Machine Learning")
st.markdown("Aplicación para demostrar las etapas de **preprocesamiento de datos** con Pandas y Scikit-learn.")

menu = st.sidebar.radio(
    "Selecciona un ejercicio:",
    ["Ejercicio 1 - Titanic", "Ejercicio 2 - Student Performance", "Ejercicio 3 - Iris"]
)

# ------------------------------------
# 🧩 EJERCICIO 1: TITANIC
# ------------------------------------
if menu == "Ejercicio 1 - Titanic":
    st.header("🚢 Ejercicio 1: Análisis del Dataset Titanic")
    st.markdown("**Objetivo:** Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.")

    file = st.file_uploader("Sube el archivo titanic.csv", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.subheader("🔹 Exploración inicial")
        st.write(df.head())
        st.write(df.info())
        st.write(df.describe())

        # Limpieza
        st.subheader("🧹 Limpieza de datos")
        df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

        st.write("Valores nulos antes del tratamiento:")
        st.write(df.isnull().sum())

        df["Age"].fillna(df["Age"].mean(), inplace=True)
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

        st.write("Valores nulos después del tratamiento:")
        st.write(df.isnull().sum())

        # Codificación
        st.subheader("🔠 Codificación de variables")
        df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

        # Estandarización
        st.subheader("⚖️ Estandarización de variables numéricas")
        scaler = StandardScaler()
        df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

        # División
        st.subheader("✂️ División en entrenamiento y prueba")
        X = df.drop("Survived", axis=1)
        y = df["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.write("Shape del conjunto de entrenamiento:", X_train.shape)
        st.write("Shape del conjunto de prueba:", X_test.shape)

        st.subheader("✅ Datos procesados (primeros 5 registros)")
        st.dataframe(df.head())

# ------------------------------------
# 🧩 EJERCICIO 2: STUDENT PERFORMANCE
# ------------------------------------
elif menu == "Ejercicio 2 - Student Performance":
    st.header("🎓 Ejercicio 2: Dataset Student Performance")
    st.markdown("**Objetivo:** Procesar los datos para un modelo que prediga la nota final (G3).")

    file = st.file_uploader("Sube el archivo student-mat.csv", type=["csv"])
    if file:
        df = pd.read_csv(file, sep=";")

        st.subheader("🔍 Exploración inicial")
        st.write(df.head())
        st.write(df.info())

        # Limpieza
        st.subheader("🧹 Limpieza de datos")
        df.drop_duplicates(inplace=True)
        st.write("Duplicados eliminados:", df.duplicated().sum())

        # Codificación categóricas
        st.subheader("🔠 One Hot Encoding de variables categóricas")
        cat_cols = df.select_dtypes(include="object").columns
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Normalización
        st.subheader("⚖️ Normalización de variables numéricas")
        num_cols = ["age", "absences", "G1", "G2"]
        scaler = MinMaxScaler()
        df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

        # División
        st.subheader("✂️ División en entrenamiento y prueba")
        X = df_encoded.drop("G3", axis=1)
        y = df_encoded["G3"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("Shape entrenamiento:", X_train.shape)
        st.write("Shape prueba:", X_test.shape)

        # Correlación
        st.subheader("📈 Correlación entre notas G1, G2 y G3")
        corr = df[["G1", "G2", "G3"]].corr()
        st.write(corr)

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("✅ Datos procesados (primeros 5 registros)")
        st.dataframe(df_encoded.head())

# ------------------------------------
# 🧩 EJERCICIO 3: IRIS
# ------------------------------------
elif menu == "Ejercicio 3 - Iris":
    st.header("🌸 Ejercicio 3: Dataset Iris")
    st.markdown("**Objetivo:** Implementar un flujo completo de preprocesamiento y visualizar resultados.")

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    st.subheader("🔍 Exploración inicial")
    st.write(df.head())
    st.write(df.describe())

    # Estandarización
    st.subheader("⚖️ Estandarización con StandardScaler")
    scaler = StandardScaler()
    df[data.feature_names] = scaler.fit_transform(df[data.feature_names])

    # División
    st.subheader("✂️ División del dataset")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write("Shape entrenamiento:", X_train.shape)
    st.write("Shape prueba:", X_test.shape)

    # Visualización
    st.subheader("🌼 Gráfico de dispersión (Sepal Length vs Petal Length)")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[data.feature_names[0]], y=df[data.feature_names[2]], hue=df["target"], palette="viridis", ax=ax)
    plt.xlabel("Sepal Length (estandarizado)")
    plt.ylabel("Petal Length (estandarizado)")
    st.pyplot(fig)

    st.subheader("📊 Estadísticas descriptivas del dataset estandarizado")
    st.write(df.describe())
