import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import plotly.express as px


partes = []

num_partes = 6

# Leer cada parte y almacenarla en la lista
for i in range(1, num_partes + 1):
    parte_df = pd.read_csv(f'df_EDA{i}.csv')
    partes.append(parte_df)

# Concatenar todos los DataFrames en uno solo
df = pd.concat(partes, ignore_index=True)
df['PrimaryDiagnosisCodePrincipal'] = df['PrimaryDiagnosisCode'].str.split('.').str[0]


#def cargar_datos():
#df = pd.read_csv('df_EDA.csv', delimiter=',', on_bad_lines='skip', engine='python')
#df['PrimaryDiagnosisCodePrincipal'] = df['PrimaryDiagnosisCode'].str.split('.').str[0]
#return df

def capitulos_genero(df):
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
        #height=800,
        #width=1000,
        hovermode='closest'
    )

    st.plotly_chart(fig)

def capitulos_edad(df):
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
    ax.set_title('Frecuencia de PrimaryDiagnosisChapter por Rangos de Edad')
    ax.set_ylabel('Rango de Edad')
    ax.set_xlabel('Frecuencia')
    ax.legend(title='PrimaryDiagnosisChapter', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

def cap2_genero_civil(df):
    filtered_df = df[df['PrimaryDiagnosisChapter'] == 2]
    frequency_df = filtered_df.groupby(['PatientGender', 'PatientMaritalStatus']).size().reset_index(name='Count')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=frequency_df, x='PatientGender', y='Count', hue='PatientMaritalStatus', ax=ax)
    ax.set_title('Frecuencia absoluta de Estado Civil del Paciente por Género del Paciente (PrimaryDiagnosisChapter = 2)')
    ax.set_xlabel('Género del Paciente')
    ax.set_ylabel('Frecuencia')
    ax.legend(title='Estado Civil del Paciente')
    st.pyplot(fig)

def frecuencia_cap_genero(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.countplot(x='PrimaryDiagnosisChapter', hue='PatientGender', data=df, palette='muted', ax=ax)
    ax.set_title('Frecuencia de Diagnósticos por Género')
    ax.set_xlabel('Capítulo de Diagnóstico')
    ax.set_ylabel('Frecuencia')
    ax.legend(title='Género del Paciente')
    st.pyplot(fig)

def frecuencia_cap_raza(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.countplot(x='PrimaryDiagnosisChapter', hue='PatientRace', data=df, palette='muted', ax=ax)
    ax.set_title('Frecuencia de Diagnósticos por Raza')
    ax.set_xlabel('Capítulo de Diagnóstico')
    ax.set_ylabel('Frecuencia')
    ax.legend(title='Raza del Paciente')
    st.pyplot(fig)

def subplot_frecuencia_cap_raza_civil_gen(df):
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
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    st.pyplot(g.fig)

def box_plot_edad_cap_civil(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='PrimaryDiagnosisChapter', y='Edad', hue='PatientMaritalStatus', data=df, palette='muted', ax=ax)
    ax.set_title('Distribución de Edad por Capítulo de Diagnóstico y Estado Civil')
    ax.set_xlabel('Capítulo de Diagnóstico')
    ax.set_ylabel('Edad')
    ax.legend(title='Estado Civil del Paciente')
    st.pyplot(fig)

def bar_plot_diagnosis_code(df):
    df_filtered = df[df['PrimaryDiagnosisChapter'] == 2]
    df_filtered['PrimaryDiagnosisCodePrincipal'] = df_filtered['PrimaryDiagnosisCode'].str.split('.').str[0]
    code_counts = df_filtered['PrimaryDiagnosisCodePrincipal'].value_counts()
    df_code_counts = code_counts.reset_index()
    df_code_counts.columns = ['PrimaryDiagnosisCodePrincipal', 'Frequency']

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_code_counts, x='PrimaryDiagnosisCodePrincipal', y='Frequency', palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Distribución de PrimaryDiagnosisCodePrincipal Capítulo 2: Neoplasia, tumores y cáncer')
    ax.set_xlabel('PrimaryDiagnosisCodePrincipal')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

def top10_cap2(df):
    df_filtered = df[df['PrimaryDiagnosisChapter'] == 2]
    df_filtered['PrimaryDiagnosisCodePrincipal'] = df_filtered['PrimaryDiagnosisCode'].str.split('.').str[0]
    code_counts = df_filtered['PrimaryDiagnosisCodePrincipal'].value_counts()
    top_10_codes = code_counts.head(10)
    df_top_10_code_counts = top_10_codes.reset_index()
    df_top_10_code_counts.columns = ['PrimaryDiagnosisCodePrincipal', 'Frequency']

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_top_10_code_counts, x='PrimaryDiagnosisCodePrincipal', y='Frequency', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('10 Códigos de Diagnóstico Principal Más Frecuentes - Capítulo 2: Neoplasia, Tumores y Cáncer')
    ax.set_xlabel('PrimaryDiagnosisCodePrincipal')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

def line_plot_admissions_over_time(df):
    df['AdmissionStartDate'] = pd.to_datetime(df['AdmissionStartDate'], format='%Y-%m-%d %H:%M:%S.%f')
    df_c92 = df[df['PrimaryDiagnosisCodePrincipal'] == 'C92']
    df_c92['AdmissionStartDate'] = df_c92['AdmissionStartDate'].dt.to_period('M')
    admissions_per_month = df_c92.groupby('AdmissionStartDate').size()

    fig, ax = plt.subplots(figsize=(12, 6))
    admissions_per_month.plot(kind='line', marker='o', linestyle='-', ax=ax)
    ax.set_title('Evolución del Número de Admisiones con Diagnóstico C92 a lo Largo del Tiempo')
    ax.set_xlabel('Fecha (Año-Mes)')
    ax.set_ylabel('Número de Admisiones')
    ax.grid(True)
    st.pyplot(fig)

def leucemia_tiempo(df):
    df['AdmissionStartDate'] = pd.to_datetime(df['AdmissionStartDate'], format='%Y-%m-%d %H:%M:%S.%f')
    df_c92 = df[df['PrimaryDiagnosisCodePrincipal'] == 'C92']
    df_c92['AdmissionStartDate'] = df_c92['AdmissionStartDate'].dt.to_period('M')
    admissions_per_month = df_c92.groupby('AdmissionStartDate').size().reset_index(name='Number of Admissions')
    admissions_per_month['AdmissionStartDate'] = admissions_per_month['AdmissionStartDate'].dt.to_timestamp()

    fig = px.bar(
        admissions_per_month,
        x='AdmissionStartDate',
        y='Number of Admissions',
        title='Evolución del Número de Admisiones con Diagnóstico C92 (Leucemia) a lo Largo del Tiempo',
        labels={'AdmissionStartDate': 'Fecha (Año-Mes)', 'Number of Admissions': 'Número de Admisiones'},
    )
    st.plotly_chart(fig)

def line_plot_admissions_trend(df):
    df['AdmissionStartDate'] = pd.to_datetime(df['AdmissionStartDate'], format='%Y-%m-%d %H:%M:%S.%f')
    df_c92 = df[df['PrimaryDiagnosisCodePrincipal'] == 'C92'].copy()
    df_c92['AdmissionStartDate'] = df_c92['AdmissionStartDate'].dt.to_period('M')
    admissions_per_month = df_c92.groupby('AdmissionStartDate').size().reset_index(name='Number of Admissions')
    admissions_per_month['AdmissionStartDate'] = admissions_per_month['AdmissionStartDate'].dt.to_timestamp()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=admissions_per_month['AdmissionStartDate'],
        y=admissions_per_month['Number of Admissions'],
        mode='lines+markers',
        name='Número de Admisiones',
        line=dict(color='darkgreen'),
        marker=dict(size=6)
    ))

    x = np.arange(len(admissions_per_month))
    y = admissions_per_month['Number of Admissions']
    lowess = sm.nonparametric.lowess(y, x, frac=0.2)

    fig.add_trace(go.Scatter(
        x=admissions_per_month['AdmissionStartDate'],
        y=lowess[:, 1],
        mode='lines',
        name='Tendencia (LOWESS)',
        line=dict(color='yellow')
    ))

    fig.update_layout(
        title='Tendencia del Número de Admisiones con Diagnóstico C92 (Leucemia) a lo Largo del Tiempo',
        xaxis_title='Fecha (Año-Mes)',
        yaxis_title='Número de Admisiones',
        template='seaborn',
        xaxis=dict(tickformat='%Y-%m')
    )

    st.plotly_chart(fig)

def main():
    st.title('Análisis de Diagnósticos por Género')
    capitulos_genero(df)
    st.write('Análisis de Diagnósticos ')
    capitulos_edad(df)
    st.write('Análisis de ')
    cap2_genero_civil(df)
    st.write('Análisis ')
    frecuencia_cap_genero(df)
    frecuencia_cap_raza(df)
    subplot_frecuencia_cap_raza_civil_gen(df)
    box_plot_edad_cap_civil(df)
    bar_plot_diagnosis_code(df)
    top10_cap2(df)
    line_plot_admissions_over_time(df)
    leucemia_tiempo(df)
    line_plot_admissions_trend(df)


if __name__ == "__main__":
    main()