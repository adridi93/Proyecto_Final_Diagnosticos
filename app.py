import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gdown
import plotly.express as px

def load_data():
    try:
        df = pd.read_csv('df_EDA.csv', delimiter=',', on_bad_lines='skip', engine='python')
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None
def main():
    st.title("Análisis de Datos de Salud")

    # Cargar los datos
    df = load_data()

    # 1. Visión general de los datos
    st.header("1. Visión general de los datos")
    st.write("Dimensiones del dataframe:", df.shape)
    st.write("Tipos de datos:", df.dtypes)

    # 2. Análisis estadístico descriptivo
    st.header("2. Análisis estadístico descriptivo")
    st.write("Estadísticas descriptivas:", df.describe())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        st.write(f"Distribución de {col}:", df[col].value_counts(normalize=True))

    # 3. Visualización interactiva
    st.header("3. Visualización interactiva")
    
    # Selección de tipo de gráfico
    chart_type = st.selectbox("Selecciona el tipo de gráfico", 
                              ["Histograma", "Scatter Plot", "Box Plot"])
    
    # Selección de columnas
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if chart_type == "Histograma":
        column = st.selectbox("Selecciona una columna para el histograma", numeric_columns)
        fig = px.histogram(df, x=column, title=f'Distribución de {column}')
        st.plotly_chart(fig)
    
    elif chart_type == "Scatter Plot":
        x_column = st.selectbox("Selecciona la columna para el eje X", numeric_columns)
        y_column = st.selectbox("Selecciona la columna para el eje Y", numeric_columns)
        fig = px.scatter(df, x=x_column, y=y_column, title=f'{x_column} vs {y_column}')
        st.plotly_chart(fig)
    
    elif chart_type == "Box Plot":
        column = st.selectbox("Selecciona una columna para el box plot", numeric_columns)
        category_columns = df.select_dtypes(include=['object']).columns.tolist()
        group_by = st.selectbox("Agrupar por (opcional)", ["Ninguno"] + category_columns)
        if group_by != "Ninguno":
            fig = px.box(df, x=group_by, y=column, title=f'Box Plot de {column} agrupado por {group_by}')
        else:
            fig = px.box(df, y=column, title=f'Box Plot de {column}')
        st.plotly_chart(fig)

    # 4. Análisis temporal
    st.header("5. Análisis temporal")
    df['AdmissionStartDate'] = pd.to_datetime(df['AdmissionStartDate'])
    df['AdmissionEndDate'] = pd.to_datetime(df['AdmissionEndDate'])
    
    fig, ax = plt.subplots()
    sns.histplot(df['AdmissionDuration'], ax=ax)
    ax.set_title('Distribución de la Duración de Admisión')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(df['TimeSinceLastAdmission'], ax=ax)
    ax.set_title('Distribución del Tiempo desde la Última Admisión')
    st.pyplot(fig)

    # 5. Análisis demográfico
    st.header("6. Análisis demográfico")
    fig, ax = plt.subplots()
    sns.histplot(df['Edad'], ax=ax)
    ax.set_title('Distribución de Edad')
    st.pyplot(fig)

    for col in ['PatientGender_Female', 'PatientGender_Male', 'PatientRace', 'PatientLanguage']:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Distribución de {col}')
        st.pyplot(fig)

    # 6. Análisis de diagnósticos
    st.header("7. Análisis de diagnósticos")
    fig, ax = plt.subplots(figsize=(12, 6))
    df['PrimaryDiagnosisChapter'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribución de Capítulos de Diagnóstico Primario')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # 7. Análisis de resultados de laboratorio
    st.header("8. Análisis de resultados de laboratorio")
    lab_cols = [col for col in df.columns if col.startswith(('CBC:', 'METABOLIC:', 'URINALYSIS:'))]
    for col in lab_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x='PrimaryDiagnosisChapter', y=col, data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f'{col} por Capítulo de Diagnóstico Primario')
        st.pyplot(fig)

    # 8. Detección de valores atípicos
    st.header("9. Detección de valores atípicos")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(df[col], ax=ax)
        ax.set_title(f'Box Plot de {col}')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
