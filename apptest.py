import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import plotly.express as px


def cargar_datos():
    df = pd.read_csv('df_EDA.csv', delimiter=',', on_bad_lines='skip', engine='python')
    df['PrimaryDiagnosisCodePrincipal'] = df['PrimaryDiagnosisCode'].str.split('.').str[0]
    return df

def scatter_plot_gender_frequency(df):
    # Crear un gráfico de dispersión de frecuencia por género
    grouped = df.groupby(['PatientGender', 'PrimaryDiagnosisChapter']).size().unstack(fill_value=0)
    female_values = grouped.loc['Female'] if 'Female' in grouped.index else pd.Series()
    male_values = grouped.loc['Male'] if 'Male' in grouped.index else pd.Series()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=female_values,
        y=male_values,
        mode='markers',
        marker=dict(
            size=15,
            color=female_values + male_values,
            colorscale='Viridis',
            showscale=True
        ),
        text=female_values.index,
        hovertemplate='<b>Código: %{text}</b><br>Mujeres: %{x}<br>Hombres: %{y}<extra></extra>'
    ))

    max_value = max(grouped.max().max(), 1)
    fig.add_trace(go.Scatter(
        x=[0, max_value],
        y=[0, max_value],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Línea de igualdad'
    ))

    fig.update_layout(
        title='PrimaryDiagnosisChapter: Frecuencia por Género',
        xaxis_title='Frecuencia en Mujeres',
        yaxis_title='Frecuencia en Hombres',
        height=800,
        width=1000,
        hovermode='closest'
    )

    st.plotly_chart(fig)
def stacked_bar_age_diagnosis(df):
        def asignar_rango_edad(edad):
            inicio = (edad // 10) * 10
            return f"{inicio:03d}-{inicio + 9:03d}"
    
        df['Rango_Edad'] = df['Edad'].apply(asignar_rango_edad)
        grouped = df.groupby(['Rango_Edad', 'PrimaryDiagnosisChapter']).size().unstack(fill_value=0)
        grouped = grouped.sort_index()
        grouped.columns = pd.to_numeric(grouped.columns, errors='coerce')
        grouped = grouped.reindex(columns=sorted(grouped.columns))
        total_freq = grouped.sum().sort_values(ascending=False)
        grouped = grouped[total_freq.index]
    
        fig, ax = plt.subplots(figsize=(14, 8))
        grouped.plot(kind='bar', stacked=True, ax=ax)
        plt.title('Frecuencia de PrimaryDiagnosisChapter por Rangos de Edad')
        plt.ylabel('Rango de Edad')
        plt.xlabel('Frecuencia')
        plt.legend(title='PrimaryDiagnosisChapter', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot()
def stacked_bar_marital_status_gender(df):
        filtered_df = df[df['PrimaryDiagnosisChapter'] == 2]
        frequency_df = filtered_df.groupby(['PatientGender', 'PatientMaritalStatus']).size().reset_index(name='Count')
    
        plt.figure(figsize=(10, 6))
        sns.barplot(data=frequency_df, x='PatientGender', y='Count', hue='PatientMaritalStatus')
        plt.title('Frecuencia absoluta de Estado Civil del Paciente por Género del Paciente (PrimaryDiagnosisChapter = 2)')
        plt.xlabel('Género del Paciente')
        plt.ylabel('Frecuencia')
        plt.legend(title='Estado Civil del Paciente')
        st.pyplot()
def count_plot_diagnosis_gender(df):
        plt.figure(figsize=(14, 8))
        sns.countplot(x='PrimaryDiagnosisChapter', hue='PatientGender', data=df, palette='muted')
        plt.title('Frecuencia de Diagnósticos por Género')
        plt.xlabel('Capítulo de Diagnóstico')
        plt.ylabel('Frecuencia')
        plt.legend(title='Género del Paciente')
        st.pyplot()
        
def count_plot_diagnosis_race(df):
        plt.figure(figsize=(14, 8))
        sns.countplot(x='PrimaryDiagnosisChapter', hue='PatientRace', data=df, palette='muted')
        plt.title('Frecuencia de Diagnósticos por Raza')
        plt.xlabel('Capítulo de Diagnóstico')
        plt.ylabel('Frecuencia')
        plt.legend(title='Raza del Paciente')
        st.pyplot()
        
def cat_plot_marital_status_gender_race(df):
        filtered_df = df[df['PrimaryDiagnosisChapter'] == 2]
        frequency_df = filtered_df.groupby(['PatientGender', 'PatientMaritalStatus', 'PatientRace']).size().reset_index(name='Count')
    
        g = sns.catplot(
            data=frequency_df, 
            x='PatientGender', 
            y='Count', 
            hue='PatientMaritalStatus', 
            col='PatientRace', 
            kind='bar', 
            height=4, 
            aspect=0.7
        )
    
        g.set_axis_labels("Género del Paciente", "Frecuencia")
        g.set_titles("{col_name}")
        g.despine(left=True)
        st.pyplot()
        
def box_plot_age_diagnosis_marital(df):
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='PrimaryDiagnosisChapter', y='Edad', hue='PatientMaritalStatus', data=df, palette='muted')
        plt.title('Distribución de Edad por Capítulo de Diagnóstico y Estado Civil')
        plt.xlabel('Capítulo de Diagnóstico')
        plt.ylabel('Edad')
        plt.legend(title='Estado Civil del Paciente')
        st.pyplot()
        
def bar_plot_diagnosis_code(df):
        df_filtered = df[df['PrimaryDiagnosisChapter'] == 2]
        df_filtered['PrimaryDiagnosisCodePrincipal'] = df_filtered['PrimaryDiagnosisCode'].str.split('.').str[0]
        code_counts = df_filtered['PrimaryDiagnosisCodePrincipal'].value_counts()
        df_code_counts = code_counts.reset_index()
        df_code_counts.columns = ['PrimaryDiagnosisCodePrincipal', 'Frequency']
    
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_code_counts, x='PrimaryDiagnosisCodePrincipal', y='Frequency', palette='viridis')
        plt.xticks(rotation=90)
        plt.title('Distribución de PrimaryDiagnosisCodePrincipal Capítulo 2: Neoplasia, tumores y cáncer')
        plt.xlabel('PrimaryDiagnosisCodePrincipal')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        st.pyplot()

def main():
    st.title('Análisis de Datos de Diagnósticos')

    # Cargar los datos
    df = cargar_datos()

    stacked_bar_age_diagnosis(df)
    scatter_plot_gender_frequency(df)


if __name__ == "__main__":
    main()

