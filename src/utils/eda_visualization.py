import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

def eda_datos_globales(df):
    st.header("Análisis Exploratorio de Datos Globales")

    col1, col2 = st.columns(2)

    with col1:
        # 1. HEATMAP DE CORRELACIÓN (Principal)
        st.subheader("Matriz de Correlación General")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
    
    with col2:
        st.subheader("Distribución de Edades por Tiempo")
        
        años_disponibles = sorted(df['Year'].unique().tolist())
        opciones_año = ["Todos"] + [str(a) for a in años_disponibles]
        año_sel = st.selectbox("Selecciona el año para el histograma de edad:", opciones_año)

        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        if año_sel == "Todos":
            sns.histplot(data=df, x="Age", hue="Year", palette="viridis", multiple="stack", ax=ax_hist)
            ax_hist.set_title("Distribución de Edades: Todos los años (Apilados)")
        else:
            df_año = df[df['Year'] == int(año_sel)]
            sns.histplot(data=df_año, x="Age", color="skyblue", kde=True, ax=ax_hist)
            ax_hist.set_title(f"Distribución de Edades en el año {año_sel}")
        
        st.pyplot(fig_hist)

    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Género por País")
        pais_sel = st.selectbox("País para el gráfico de queso:", df['Country_Region'].unique())
        df_pais = df[df['Country_Region'] == pais_sel]
        gender_counts = df_pais['Gender'].value_counts()
        
        if not gender_counts.empty:
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
                       colors=sns.color_palette("pastel"), startangle=140)
            st.pyplot(fig_pie)

    with col2:
        st.subheader("Edad por Región")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df, x="Age", y="Country_Region", palette="Set2", ax=ax_box)
        st.pyplot(fig_box)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución Demográfica por Ciudad y Contaminación")
        
        col_a, col_b = st.columns(2)
        with col_a:
            ciudad_sel = st.selectbox("Selecciona Ciudad/País:", df['Country_Region'].unique(), key="city_pie")
        with col_b:
            # Filtramos un rango de contaminación para ver el impacto en ese grupo
            nivel_aire = st.slider("Filtrar por nivel mínimo de Contaminación:", 
                                float(df['Air_Pollution'].min()), float(df['Air_Pollution'].max()), 0.0)

        df_filtered = df[(df['Country_Region'] == ciudad_sel) & (df['Air_Pollution'] >= nivel_aire)]

        if not df_filtered.empty:
            fig_q4, ax_q4 = plt.subplots(figsize=(8, 6))
            # Creamos una columna combinada para el queso: Género + Rango de Edad
            df_filtered['Demografico'] = df_filtered['Gender'] + " (" + pd.cut(df_filtered['Age'], bins=[0,40,60,100], labels=['Joven','Adulto','Senior']).astype(str) + ")"
            counts = df_filtered['Demografico'].value_counts()
            
            ax_q4.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette("magma"), startangle=140)
            ax_q4.set_title(f"Demografía en {ciudad_sel} con Aire >= {nivel_aire}")
            st.pyplot(fig_q4)
        else:
            st.warning("No hay datos para esos filtros de contaminación en esta ciudad.")
    
    with col2:
        st.subheader("Composición de Factores de Riesgo por País")
        
        pais_riesgo = st.selectbox("Selecciona un país para ver sus riesgos:", 
                                df['Country_Region'].unique(), key="riesgo_pais")
        
        factores = ['Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
        df_pais_riesgo = df[df['Country_Region'] == pais_riesgo][factores].mean()

        fig_riesgo, ax_riesgo = plt.subplots(figsize=(7, 7))
        ax_riesgo.pie(df_pais_riesgo, labels=factores, autopct='%1.1f%%', 
                    colors=sns.color_palette("Set2"), startangle=140, 
                    explode=[0.05]*4, shadow=True)
        ax_riesgo.set_title(f"Distribución Media de Riesgos en {pais_riesgo}")
        st.pyplot(fig_riesgo)
    
    st.divider()

    # 1. HISTOGRAMA DE FACTORES DE RIESGO POR PAÍS (PORCENTAJE)
    st.subheader("Composición de Factores de Riesgo por País")
    
    pais_riesgo = st.selectbox("Selecciona un país para ver sus riesgos:", 
                               df['Country_Region'].unique(), key="riesgo_pais_bar")
    
    factores = ['Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
    
    # Calculamos la media de cada factor para ese país
    df_pais_mean = df[df['Country_Region'] == pais_riesgo][factores].mean()
    
    # Convertimos a porcentaje relativo (cada barra será su peso sobre el total de riesgos)
    total_riesgo = df_pais_mean.sum()
    df_pais_pct = (df_pais_mean / total_riesgo) * 100

    fig_riesgo, ax_riesgo = plt.subplots(figsize=(9, 6))
    sns.barplot(x=df_pais_pct.index, y=df_pais_pct.values, palette="viridis", ax=ax_riesgo)

    # Añadir etiquetas de porcentaje sobre las barras
    for i, p in enumerate(ax_riesgo.patches):
        ax_riesgo.annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 9), 
                           textcoords='offset points', fontsize=11, fontweight='bold')

    ax_riesgo.set_title(f"Distribución Porcentual de Riesgos en {pais_riesgo}")
    ax_riesgo.set_ylabel("Porcentaje (%)")
    ax_riesgo.set_ylim(0, df_pais_pct.max() + 10) # Espacio para la etiqueta
    st.pyplot(fig_riesgo)

def eda_datos_sinteticos(df):
    st.header("Análisis Exploratorio: Simulador de Pacientes")

    # 1. MATRIZ DE CORRELACIÓN Y DIAGNÓSTICO
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlación de Factores")
        # Filtramos solo numéricas y quitamos ID si existe
        df_num = df.select_dtypes(include=['number'])
        if not df_num.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
            st.pyplot(fig_corr)

    with col2:
        st.subheader("Diagnóstico por Género")
        # Ver cuántos positivos/negativos hay por género
        fig_diag, ax_diag = plt.subplots(figsize=(10, 8))
        sns.countplot(data=df, x='Genero', hue='Diagnostico', palette='viridis', ax=ax_diag)
        ax_diag.set_title("Frecuencia de Diagnóstico (0=Sano, 1=Cáncer)")
        st.pyplot(fig_diag)

    st.divider()

    # 2. EDAD Y HÁBITOS
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribución de Edad")
        # Selector para ver sanos, enfermos o todos
        estado = st.radio("Mostrar por estado:", ["Todos", "Sanos (0)", "Positivos (1)"], horizontal=True)
        
        fig_age, ax_age = plt.subplots(figsize=(10, 6))
        if estado == "Todos":
            sns.histplot(data=df, x="Edad", hue="Diagnostico", kde=True, element="step", ax=ax_age)
        else:
            val = 0 if "Sanos" in estado else 1
            sns.histplot(df[df['Diagnostico'] == val]['Edad'], kde=True, color="skyblue", ax=ax_age)
        
        st.pyplot(fig_age)

    with col4:
        st.subheader("Riesgo Genético por Género")
        # Usamos barplot para ver el porcentaje/media de riesgo hereditario
        fig_gen, ax_gen = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Genero", y="Componente_Hereditario", palette="magma", ax=ax_gen)
        ax_gen.set_ylabel("Media de Riesgo Hereditario")
        st.pyplot(fig_gen)

    st.divider()

    # 3. FACTORES DE RIESGO (ESTILO HISTOGRAMA PORCENTUAL)
    st.subheader("Peso de Factores de Riesgo en el Simulador")
    
    # Seleccionamos factores clave de tu CSV
    factores_clave = ['Fumador', 'Consume_Alcohol', 'Sedentarismo', 'Obesidad', 'Diabetes_tipo_2']
    
    # Calculamos la media (que en binario 0/1 es equivalente a la frecuencia)
    df_factores = df[factores_clave].mean() * 100
    
    fig_factores, ax_f = plt.subplots(figsize=(12, 6))
    sns.barplot(x=df_factores.index, y=df_factores.values, palette="rocket", ax=ax_f)
    
    # Anotar porcentajes
    for p in ax_f.patches:
        ax_f.annotate(f'{p.get_height():.1f}%', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', xytext=(0, 9), 
                      textcoords='offset points', fontweight='bold')

    ax_f.set_ylabel("Prevalencia en Pacientes (%)")
    ax_f.set_title("¿Qué tan comunes son estos factores en los datos simulados?")
    st.pyplot(fig_factores)

    st.divider()

    # 4. MARCADOR TUMORAL (CEA) VS DIAGNÓSTICO
    st.subheader("Marcador CEA vs Diagnóstico")
    col5, col6 = st.columns(2)

    with col5:
        # Relación entre el nivel de CEA y si tiene cáncer
        fig_cea, ax_cea = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="Diagnostico", y="CEA_Level_ng_mL (Marcador Tumoral)", palette="Set2", ax=ax_cea)
        ax_cea.set_title("Niveles de CEA (Marcador Tumoral)")
        st.pyplot(fig_cea)

    with col6:
        # Resultado de Sangre en Heces (FOBT)
        st.write("**Resultado FOBT (Sangre en Heces)**")
        fobt_counts = df['FOBT_Resultado (Sangre en heces)'].value_counts()
        fig_fobt, ax_fobt = plt.subplots()
        ax_fobt.pie(fobt_counts, labels=fobt_counts.index, autopct='%1.1f%%', 
                    colors=['#ff9999','#66b3ff'], startangle=90, wedgeprops={'edgecolor': 'white'})
        st.pyplot(fig_fobt)

def eda_datos_combinados(df):
    st.header("Análisis Exploratorio: Datos Combinados")

    # --- BLOQUE 1: ANALISIS DE COMPORTAMIENTO DE RIESGO ---
    st.subheader("1. Mapa de Calor de Correlación Integrada")

    # --- ARREGLO PARA EL ERROR DE COLUMNAS ---
    # Si Gender_n no existe, lo creamos al vuelo para el gráfico
    if 'Gender_n' not in df.columns and 'Gender' in df.columns:
        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        df['Gender_n'] = df['Gender'].map(gender_map).fillna(0)

    # Seleccionamos solo las variables que tienen relevancia clínica en ambos sets
    cols_estudio = [
        'Age', 
        'Gender_n',        # Asegúrate de usar la versión con _n
        'Smoking', 
        'Alcohol_Use', 
        'Obesity_Level', 
        'Genetic_Risk', 
        'Target_Severity_Score'
    ]
    # Y luego calcular la correlación
    matriz_corr = df[cols_estudio].corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
    st.caption("Interpretación: Los valores cercanos a 1 indican una relación fuerte entre el hábito y la severidad/diagnóstico.")

    st.divider()

    # --- BLOQUE 2: COMPARATIVA POR ORIGEN DE DATOS ---
    # Creamos una columna temporal para identificar de dónde viene cada fila
    # (El simulador no tiene 'Country_Region', así que lo usamos para distinguir)
    df['Origen'] = df['Country_Region'].apply(lambda x: 'Global' if pd.notnull(x) and x != 0 else 'Simulador')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Distribución de Severidad por Origen")
        fig_sev, ax_sev = plt.subplots()
        sns.kdeplot(data=df, x="Target_Severity_Score", hue="Origen", fill=True, common_norm=False, ax=ax_sev)
        ax_sev.set_title("Simulador vs Realidad Global")
        st.pyplot(fig_sev)
        st.write("¿El simulador está sobreestimando la gravedad de los casos?")

    with col2:
        st.subheader("3. Perfil de Edad y Estilo de Vida")
        # Gráfico de burbujas: Edad vs Severidad, tamaño por Obesidad
        fig_scat, ax_scat = plt.subplots()
        sample_df = df.sample(n=min(1000, len(df))) # Muestreo para no saturar el gráfico
        sns.scatterplot(data=sample_df, x="Age", y="Target_Severity_Score", 
                        hue="Origen", size="Obesity_Level", alpha=0.5, ax=ax_scat)
        st.pyplot(fig_scat)

    st.divider()

    # --- BLOQUE 3: IMPACTO DE LOS FACTORES CLÍNICOS ---
    st.subheader("4. Análisis de Factores de Riesgo por Grupos de Edad")
    
    # Creamos rangos de edad para un análisis más "Data Analyst"
    df['Rango_Edad'] = pd.cut(df['Age'], bins=[0, 18, 40, 65, 100], 
                             labels=['Menor', 'Adulto Joven', 'Adulto', 'Senior'])
    
    # Derretimos el dataframe para poder comparar Smoking vs Alcohol vs Obesity en un solo gráfico
    factores = ['Smoking', 'Alcohol_Use', 'Obesity_Level']
    df_melted = df.melt(id_vars=['Rango_Edad'], value_vars=factores, 
                        var_name='Factor', value_name='Nivel')
    
    fig_fact, ax_fact = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_melted, x="Factor", y="Nivel", hue="Rango_Edad", palette="Set3", ax=ax_fact)
    ax_fact.set_title("Distribución de Riesgos por Etapa de Vida")
    st.pyplot(fig_fact)

    st.divider()

    # --- BLOQUE 4: TABLA DE INSIGHTS (OPCIONAL) ---
    st.subheader("Resumen Ejecutivo de Riesgos")
    resumen = df.groupby('Origen')[factores + ['Target_Severity_Score']].mean()
    st.table(resumen)