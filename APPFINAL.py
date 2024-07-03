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
        title='Frecuencia de diagnóstico por capítulos, distribuido por género.',
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
    chapters = sorted(df['PrimaryDiagnosisChapter'].unique())
    selected_chapter = st.selectbox('Selecciona el capítulo de diagnóstico:', chapters, key='selectbox1')

    # Filtrar el DataFrame basado en la selección
    df_filtered = df[df['PrimaryDiagnosisChapter'] == selected_chapter]
    df_filtered['PrimaryDiagnosisCodePrincipal'] = df_filtered['PrimaryDiagnosisCode'].str.split('.').str[0]
    code_counts = df_filtered['PrimaryDiagnosisCodePrincipal'].value_counts()
    df_code_counts = code_counts.reset_index()
    df_code_counts.columns = ['PrimaryDiagnosisCodePrincipal', 'Frequency']

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_code_counts, x='PrimaryDiagnosisCodePrincipal', y='Frequency', palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Distribución de PrimaryDiagnosisCodePrincipal')
    ax.set_xlabel('PrimaryDiagnosisCodePrincipal')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)
def top10_capX(df):
    
    # Cargar el archivo CSV con códigos y descripciones
    codigo_descripcion = pd.read_csv('CIE10.csv', header=None, names=['Codigo', 'Descripcion'], sep='\t')

    # Crear un desplegable para seleccionar el capítulo
    chapters = sorted(df['PrimaryDiagnosisChapter'].unique())
    selected_chapter = st.selectbox('Selecciona el capítulo de diagnóstico:', chapters)

    # Filtrar el DataFrame basado en la selección
    df_filtered = df[df['PrimaryDiagnosisChapter'] == selected_chapter]

    # Procesar los códigos
    df_filtered['PrimaryDiagnosisCodePrincipal'] = df_filtered['PrimaryDiagnosisCode'].str.split('.').str[0]
    code_counts = df_filtered['PrimaryDiagnosisCodePrincipal'].value_counts()
    top_10_codes = code_counts.head(10)

    # Crear DataFrame con los top 10 códigos y sus frecuencias
    df_top_10_code_counts = top_10_codes.reset_index()
    df_top_10_code_counts.columns = ['PrimaryDiagnosisCodePrincipal', 'Frequency']

    # Unir con las descripciones
    df_top_10_code_counts = df_top_10_code_counts.merge(
        codigo_descripcion, 
        left_on='PrimaryDiagnosisCodePrincipal', 
        right_on='Descripcion', 
        how='left'
    )

    # Mostrar la tabla con códigos, frecuencias y descripciones
    st.write(df_top_10_code_counts[['PrimaryDiagnosisCodePrincipal', 'Frequency', 'Descripcion']])

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_top_10_code_counts, x='PrimaryDiagnosisCodePrincipal', y='Frequency', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'10 Códigos de Diagnóstico Principal Más Frecuentes - Capítulo {selected_chapter}')
    ax.set_xlabel('PrimaryDiagnosisCodePrincipal')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

def top10_cap2(df):
    codigo_descripcion = pd.read_csv('CIE10.csv', header=None, names=['Codigo', 'Descripcion'])
    codigo_descripcion_dict = codigo_descripcion.set_index('Codigo')['Descripcion'].to_dict()
    
    
    # Crear un desplegable para seleccionar el capítulo
    chapters = sorted(df['PrimaryDiagnosisChapter'].unique())
    selected_chapter = st.selectbox('Selecciona el capítulo de diagnóstico:', chapters, key='selectbox2')

    # Filtrar el DataFrame basado en la selección
    df_filtered = df[df['PrimaryDiagnosisChapter'] == selected_chapter]

    #df_filtered = df[df['PrimaryDiagnosisChapter'] == 2]
    df_filtered['PrimaryDiagnosisCodePrincipal'] = df_filtered['PrimaryDiagnosisCode'].str.split('.').str[0]
    code_counts = df_filtered['PrimaryDiagnosisCodePrincipal'].value_counts()
    top_10_codes = code_counts.head(10)
    # Crear df_top_10_code_counts
    df_top_10_code_counts = top_10_codes.reset_index()
    df_top_10_code_counts.columns = ['PrimaryDiagnosisCodePrincipal', 'Frequency']

    # Añadir la columna 'Descripcion'
    df_top_10_code_counts['Descripcion'] = df_top_10_code_counts['PrimaryDiagnosisCodePrincipal'].map(codigo_descripcion_dict)
    
    #valor = codigo_descripcion_dict['{selected_chapter}']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_top_10_code_counts, x='PrimaryDiagnosisCodePrincipal', y='Frequency', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('10 Códigos de Diagnóstico Principal Más Frecuentes')
    #ax.set_title(f'10 Códigos de Diagnóstico Principal Más Frecuentes - Capítulo 2: Neoplasia, Tumores y Cáncer{valor}')
    ax.set_xlabel('PrimaryDiagnosisCodePrincipal')
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

    st.dataframe(df_top_10_code_counts)

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
    st.title('ESTUDIO DIAGNÓSTICOS CLÍNICOS (CIE-10)')
    st.write('Se ha realizado un estudio de diagnósticos clínicos basado en los resultados clasificados por el método CIE-10, Clasificación Internacional de Enfermedades. El CIE-10 es un sistema de codificación que permite registrar de manera estandarizada las enfermedades y problemas de salud, facilitando la comunicación entre profesionales de la salud y la comparación de datos a nivel mundial. En este estudio, se analizan los diagnósticos clínicos utilizando esta clasificación para identificar patrones y tendencias.')
    capitulos_genero(df)
    st.write('En este gráfico se muestra la distribución de diagnósticos en los veintiún capítulos, desglosados por género. La mayoría de los capítulos presentan un número de diagnósticos por debajo de los doce mil pacientes, tanto en hombres como en mujeres, mostrando una alineación notable entre ambos géneros. Los capítulos que sobresalen son el capítulo 2, que aborda las neoplasias (tumores y cáncer) y en algo de menor impacto el capítulo 13, relacionado con enfermedades del sistema osteomuscular y del tejido conectivo. Además, podemos observar una mayor frecuencia de diagnóstico en el caso de mujeres, como también se puede ver en el siguiente gráfico:')
    frecuencia_cap_genero(df)
    st.title('Frecuencia de diagnóstico por capítulos, distribuido por edad.')
    capitulos_edad(df)
    st.write('En esta representación gráfica, donde vuelven a destacar los mismos capítulos ya mencionados, cabe destacar la disposición estadística de los diagnósticos por edad. Podemos observar una asimetría positiva, donde la media la encontramos en la franja 60-69 años, con mayores valores separados de la media cuanto mayor son los pacientes.')
    st.title('Frecuencia de diagnóstico por capítulos, distribuido por Raza, Género y Estado Civil.')
    frecuencia_cap_raza(df)
    subplot_frecuencia_cap_raza_civil_gen(df)
    st.write('De esta composición de gráficos se puede extraer conclusiones claras. Las distribuciones son simétricas en todos los capítulos, permitiendo concluir que la raza predominante en el estudio es la blanca, seguida por la asiática y, en último lugar, la afroamericana. Además, las personas casadas son las que presentan un mayor número de enfermedades diagnosticadas, seguidas de cerca por las personas solteras, y, como se mencionó anteriormente, en todos los casos las mujeres presentan cifras superiores a las de los hombres.')
    st.title('Frecuencia de diagnóstico por capítulos, distribuido por Edad y Estado Civil.')
    box_plot_edad_cap_civil(df)
    st.write('Por último, nos enfocamos en el estado civil desglosado por edad para cada uno de los capítulos. En este caso, resulta especialmente interesante centrarnos en los divorciados, ya que presentan las mayores desviaciones respecto a la media total. Destacamos el capítulo 12, enfermedades de la piel y el tejido subcutáneo, donde los divorciados sobresalen a la baja, sin presentar ningún caso por encima de los 90 años. En cambio, en los capítulos 10, enfermedades del aparato respiratorio; 16, ciertas afecciones originadas en el periodo perinatal; y 17, malformaciones congénitas, deformidades y anomalías cromosómicas, los divorciados también destacan, pero en este caso al alza, mostrando las mayores edades y, por tanto, una mayor tasa de supervivencia.')
    st.write('Como conclusión general del estudio podemos afirmar que los mayores riesgos están presentes en las mujeres, blancas, casadas o solteras, con una edad media de detección en la década de los 60. Siendo la mayor intensidad de casos localizados en el capítulo 2 Neoplasias (tumores y cáncer), así como en el capítulo 13, las enfermedades del sistema osteomuscular y del tejido conectivo, aunque ambos presentan una tasa de supervivencia positiva. En cambio, capítulos con una menor frecuencia, y por tanto, posiblemente también relacionado con una mayor mortalidad, podemos afirmar que son el capítulo 12, enfermedades de la piel y el tejido subcutáneo, el capítulo 18, Síntomas, signos y hallazgos anormales clínicos y de laboratorio, y el capítulo 20, Causas externas de morbilidad y de mortalidad, ya que son los que presentan unas franjas de edad más restringidas. Es decir, en esos casos, aunque muy escasos, la detección es más tardía y la franja de edad máxima suele ser menor en comparación con otros capítulos.')
    st.write('A continuación se muestra en detalle cada uno de los capítulos con la frecuencia de detección de cada uno de sus subcódigos, destacando los 10 más frecuentes así como la evolución del principal subcódigo a lo largo del tiempo.')
    bar_plot_diagnosis_code(df)
    top10_cap2(df)
    #top10_capX(df)
    #line_plot_admissions_over_time(df)
    leucemia_tiempo(df)

    #line_plot_admissions_trend(df)


if __name__ == "__main__":
    main()