import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns


def eda_datos_globales(df):
    st.header("Análisis exploratorio de datos globales")

    col1, col2 = st.columns(2)

    with col1:
        # --- 1. MATRIZ DE CORRELACIÓN ---
        st.subheader("Matriz de correlación general")
        # Filtramos solo las columnas numéricas para poder calcular la correlación
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if not numeric_cols.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            # Mostramos un heatmap para identificar qué variables numéricas están más relacionadas
            sns.heatmap(
                df[numeric_cols].corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax_corr,
            )
            st.pyplot(fig_corr)

    with col2:
        # --- 2. DISTRIBUCIÓN DE EDADES ---
        st.subheader("Distribución de edades por tiempo")

        # Obtenemos los años disponibles y creamos un selector para filtrar el gráfico
        años_disponibles = sorted(df["Year"].unique().tolist())
        opciones_año = ["Todos"] + [str(a) for a in años_disponibles]
        año_sel = st.selectbox(
            "Selecciona el año para el histograma de edad:", opciones_año
        )

        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        if año_sel == "Todos":
            # Si se seleccionan todos los años, mostramos un histograma apilado por año
            sns.histplot(
                data=df,
                x="Age",
                hue="Year",
                palette="viridis",
                multiple="stack",
                ax=ax_hist,
            )
            ax_hist.set_title("Distribución de edades: Todos los años (Apilados)")
        else:
            # Si se selecciona un año particular, mostramos el histograma básico de ese año
            df_año = df[df["Year"] == int(año_sel)]
            sns.histplot(data=df_año, x="Age", color="skyblue", kde=True, ax=ax_hist)
            ax_hist.set_title(f"Distribución de edades en el año {año_sel}")

        st.pyplot(fig_hist)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        # --- 3. PROPORCIÓN DE GÉNERO POR PAÍS ---
        st.subheader("Género por país")
        pais_sel = st.selectbox(
            "País para el gráfico de queso:", df["Country_Region"].unique()
        )
        df_pais = df[df["Country_Region"] == pais_sel]
        gender_counts = df_pais["Gender"].value_counts()

        if not gender_counts.empty:
            # Mostramos un gráfico de pastel (pie chart) con el porcentaje de cada género en el país seleccionado
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(
                gender_counts,
                labels=gender_counts.index,
                autopct="%1.1f%%",
                colors=sns.color_palette("pastel"),
                startangle=140,
            )
            st.pyplot(fig_pie)

    with col2:
        # --- 4. EDAD VS REGIÓN (BOXPLOT) ---
        st.subheader("Edad por región")
        fig_box, ax_box = plt.subplots()
        # Generamos un boxplot para ver la dispersión de las edades en cada región y buscar posibles valores atípicos
        sns.boxplot(data=df, x="Age", y="Country_Region", palette="Set2", ax=ax_box)
        st.pyplot(fig_box)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        # --- 5. DEMOGRAFÍA FILTRADA POR CONTAMINACIÓN ---
        st.subheader("Distribución demográfica por ciudad y contaminación")

        col_a, col_b = st.columns(2)
        with col_a:
            ciudad_sel = st.selectbox(
                "Selecciona País/Región:", df["Country_Region"].unique(), key="city_pie"
            )
        with col_b:
            # Slider para filtrar a las personas expuestas a un nivel mínimo de contaminación
            nivel_aire = st.slider(
                "Filtrar por nivel mínimo de Contaminación:",
                float(df["Air_Pollution"].min()),
                float(df["Air_Pollution"].max()),
                0.0,
            )

        # Filtramos el dataset con la selección de país y contaminación
        df_filtered = df[
            (df["Country_Region"] == ciudad_sel) & (df["Air_Pollution"] >= nivel_aire)
        ]

        if not df_filtered.empty:
            fig_q4, ax_q4 = plt.subplots(figsize=(8, 6))
            # Creamos una agrupación segmentar a la población afectada: "Género (Rango de Edad)"
            df_filtered["Demografico"] = (
                df_filtered["Gender"]
                + " ("
                + pd.cut(
                    df_filtered["Age"],
                    bins=[0, 40, 60, 100],
                    labels=["Joven", "Adulto", "Senior"],
                ).astype(str)
                + ")"
            )
            counts = df_filtered["Demografico"].value_counts()

            # Gráfico de pastel demostrando los colectivos más afectados según estos filtros
            ax_q4.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=sns.color_palette("magma"),
                startangle=140,
            )
            ax_q4.set_title(
                f"Demografía en {ciudad_sel} expuesta a Contaminación >= {nivel_aire}"
            )
            st.pyplot(fig_q4)
        else:
            st.warning("No hay datos que coincidan con estos filtros.")

    with col2:
        # --- 6. FACTORES DE RIESGO POR PAÍS (GRÁFICO PASTEL) ---
        st.subheader("Composición de factores de riesgo por país")

        pais_riesgo = st.selectbox(
            "Selecciona el país de interés:",
            df["Country_Region"].unique(),
            key="riesgo_pais",
        )

        # Variables que consideramos como factores principales de riesgo
        factores = ["Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level"]
        # Calculamos la media de cada factor de riesgo en el país elegido
        df_pais_riesgo = df[df["Country_Region"] == pais_riesgo][factores].mean()

        fig_riesgo, ax_riesgo = plt.subplots(figsize=(7, 7))
        ax_riesgo.pie(
            df_pais_riesgo,
            labels=factores,
            autopct="%1.1f%%",
            colors=sns.color_palette("Set2"),
            startangle=140,
            explode=[0.05] * 4,
            shadow=True,
        )
        ax_riesgo.set_title(f"Distribución Media de Riesgos en {pais_riesgo}")
        st.pyplot(fig_riesgo)

    st.divider()

    # --- 7. FACTORES DE RIESGO (HISTOGRAMA PORCENTUAL) ---
    st.subheader("Distribución porcentual relativa de factores de riesgo")

    pais_riesgo_bar = st.selectbox(
        "Selecciona un país (Gráfico Barras):",
        df["Country_Region"].unique(),
        key="riesgo_pais_bar",
    )

    df_pais_mean = df[df["Country_Region"] == pais_riesgo_bar][factores].mean()

    # Convertimos la media a un porcentaje relativo sobre el total de factores
    # para comparar visualmente cuál de estos cuatro problemas es el más severo en el país
    total_riesgo = df_pais_mean.sum()
    df_pais_pct = (df_pais_mean / total_riesgo) * 100

    fig_riesgo_bar, ax_riesgo_bar = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x=df_pais_pct.index, y=df_pais_pct.values, palette="viridis", ax=ax_riesgo_bar
    )

    # Añadir las etiquetas de porcentaje encima de cada barra
    for i, p in enumerate(ax_riesgo_bar.patches):
        ax_riesgo_bar.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    ax_riesgo_bar.set_title(f"Peso de los diferentes riesgos en {pais_riesgo_bar}")
    ax_riesgo_bar.set_ylabel("Importancia relativa (%)")
    ax_riesgo_bar.set_ylim(
        0, df_pais_pct.max() + 10
    )  # Damos espacio extra arriba para la etiqueta
    st.pyplot(fig_riesgo_bar)


def eda_datos_sinteticos(df):
    st.header("Análisis exploratorio: datos simulados (sintéticos)")

    col1, col2 = st.columns(2)

    with col1:
        # --- 1. MATRIZ DE CORRELACIÓN SINTÉTICA ---
        st.subheader("Correlación de factores")
        # Filtramos solo numéricas y quitamos ID si existe
        df_num = df.select_dtypes(include=["number"])
        if not df_num.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            # Mostramos un mapa de calor que ayude a ver qué reglas de simulación crearon relaciones fuertes
            sns.heatmap(
                df_num.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax_corr,
                annot_kws={"size": 8},
            )
            st.pyplot(fig_corr)

    with col2:
        # --- 2. DIAGNÓSTICO VS GÉNERO ---
        st.subheader("Frecuencia de diagnóstico por género")
        # Un countplot que nos muestra cómo balanceó el simulador los pacientes enfermos vs sanos
        fig_diag, ax_diag = plt.subplots(figsize=(10, 8))
        sns.countplot(
            data=df, x="Genero", hue="Diagnostico", palette="viridis", ax=ax_diag
        )
        ax_diag.set_title("Casos Simulados (0=Sano, 1=Cáncer)")
        st.pyplot(fig_diag)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        # --- 3. EDAD Y DIAGNÓSTICO ---
        st.subheader("Distribución de edad según estado")
        estado = st.radio(
            "Filtro de visualización por estado:",
            ["Todos", "Sanos (0)", "Positivos (1)"],
            horizontal=True,
        )

        fig_age, ax_age = plt.subplots(figsize=(10, 6))
        if estado == "Todos":
            # Compara la curva de edad de sanos vs enfermos
            sns.histplot(
                data=df,
                x="Edad",
                hue="Diagnostico",
                kde=True,
                element="step",
                ax=ax_age,
            )
        else:
            val = 0 if "Sanos" in estado else 1
            sns.histplot(
                df[df["Diagnostico"] == val]["Edad"],
                kde=True,
                color="skyblue",
                ax=ax_age,
            )

        st.pyplot(fig_age)

    with col4:
        # --- 4. RIESGO HEREDITARIO ---
        st.subheader("Riesgo genético hereditario por género")
        # Usamos barplot para ver el porcentaje/media de riesgo hereditario
        fig_gen, ax_gen = plt.subplots(figsize=(10, 6))
        # Verifica si hay algún sesgo de riesgo genético artificial por género en los datos sintéticos
        sns.barplot(
            data=df, x="Genero", y="Componente_Hereditario", palette="magma", ax=ax_gen
        )
        ax_gen.set_ylabel("Valor Medio de Riesgo Hereditario")
        st.pyplot(fig_gen)

    st.divider()

    # --- 5. PREVALENCIA DE FACTORES DE RIESGO (HISTOGRAMA) ---
    st.subheader("Prevalencia de hábitos y factores de riesgo simulado")

    factores_clave = [
        "Fumador",
        "Consume_Alcohol",
        "Sedentarismo",
        "Obesidad",
        "Diabetes_tipo_2",
    ]

    # Dado que los valores simulados probablemente sean 0 (No) o 1 (Sí),
    # hacemos la media y multiplicamos por 100 para obtener el porcentaje de prevalencia en la población.
    df_factores = df[factores_clave].mean() * 100

    fig_factores, ax_f = plt.subplots(figsize=(12, 6))
    sns.barplot(x=df_factores.index, y=df_factores.values, palette="rocket", ax=ax_f)

    # Añadimos el valor porcentual encima de cada barra
    for p in ax_f.patches:
        ax_f.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontweight="bold",
        )

    ax_f.set_ylabel("Aparición en pacientes simulados (%)")
    ax_f.set_title("Frecuencia de factores de riesgo introducidos en la simulación")
    st.pyplot(fig_factores)

    st.divider()

    # --- 6. MARCADORES MÉDICOS VS DIAGNÓSTICO ---
    st.subheader("Resultados de pruebas médicas frente a diagnósticos")
    col5, col6 = st.columns(2)

    with col5:
        # Gráfico Boxplot para contrastar si el nivel de CEA (marcador tumoral) es consistentemente
        # más alto en los pacientes diagnosticados con cáncer (1) frente a los sanos (0).
        fig_cea, ax_cea = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x="Diagnostico",
            y="CEA_Level_ng_mL (Marcador Tumoral)",
            palette="Set2",
            ax=ax_cea,
        )
        ax_cea.set_title("Relación de marcador tumoral CEA con diagnóstico")
        st.pyplot(fig_cea)

    with col6:
        st.write("**Prevalencia de falsos/verdaderos en test sangre oculta (FOBT)**")
        fobt_counts = df["FOBT_Resultado (Sangre en heces)"].value_counts()
        fig_fobt, ax_fobt = plt.subplots()
        # Mostramos la proporción total de pruebas de heces positivas vs negativas generadas
        ax_fobt.pie(
            fobt_counts,
            labels=fobt_counts.index,
            autopct="%1.1f%%",
            colors=["#ff9999", "#66b3ff"],
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        st.pyplot(fig_fobt)


def eda_datos_combinados(df):
    st.header("Análisis exploratorio: datos combinados")

    # --- 1. Análisis de riesgo ---
    st.subheader("Heatmap de Correlación Integrada de Factores Médicos")

    # Adaptación: Para calcular correlaciones homogéneas, primero numerizamos
    # la columna de género solo si es categórica y no ha sido convertida aún
    if "Gender" in df.columns:
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        df["Gender_n"] = df["Gender"].map(gender_map).fillna(0)

    # Variables estandarizadas que deberían estar presentes en ambos datasets compartidos
    cols_estudio = [
        "Age",
        "Gender_n",
        "Smoking",
        "Alcohol_Use",
        "Obesity_Level",
        "Genetic_Risk",
        "Target_Severity_Score",  # la métrica resultante de severidad del caso
    ]
    # Calculamos la correlación general combinando las dos naturalezas de los datos
    matriz_corr = df[cols_estudio].corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
    st.caption(
        "Interpretación: Los valores cercanos a 1 indican una relación fuerte entre el hábito y la severidad/diagnóstico."
    )

    st.divider()

    # --- BLOQUE 2: COMPARATIVA POR ORIGEN DE DATOS ---
    # Creamos una columna temporal para identificar de dónde viene cada fila
    # (El simulador no tiene 'Country_Region', así que lo usamos para distinguir)
    df["Origen"] = df["Country_Region"].apply(
        lambda x: "Global" if pd.notnull(x) and x != 0 else "Simulador"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de Severidad por Origen")
        fig_sev, ax_sev = plt.subplots()
        # El gráfico KDE es ideal para superponer curvas y ver
        # si los datos simulados tienen el mismo perfil de severidad clínica que la realidad
        sns.kdeplot(
            data=df,
            x="Target_Severity_Score",
            hue="Origen",
            fill=True,
            common_norm=False,
            ax=ax_sev,
        )
        ax_sev.set_title("Comparativa: Simulador vs Realidad")
        st.pyplot(fig_sev)
        st.write(
            "Útil para validar si el simulador refleja de forma realista la tasa de severidad global."
        )

    with col2:
        st.subheader("3. Perfil de Edad y Estilo de Vida")
        # Gráfico de burbujas: Edad vs Severidad, tamaño por Obesidad
        fig_scat, ax_scat = plt.subplots()
        # Extraer una muestra aleatoria prevendrá un colapso visual si el registro conjunto es enorme
        sample_df = df.sample(n=min(1000, len(df)))
        # Gráfico de puntos multivariable (Edad en X, Severidad en Y, Nivel de obesidad como tamaño)
        sns.scatterplot(
            data=sample_df,
            x="Age",
            y="Target_Severity_Score",
            hue="Origen",
            size="Obesity_Level",
            alpha=0.5,
            ax=ax_scat,
        )
        st.pyplot(fig_scat)

    st.divider()

    # --- BLOQUE 3: IMPACTO DE LOS FACTORES CLÍNICOS ---
    st.subheader("4. Análisis de factores de riesgo por grupos de edad")

    # Agrupamos todas las edades brutas en rangos analíticos estandarizados
    df["Rango_Edad"] = pd.cut(
        df["Age"],
        bins=[0, 40, 65, 100],
        labels=["Adulto Joven", "Adulto", "Senior"],
    )

    # 'Derretimos' (melt) el dataset. Transformamos varias columnas de hábitos (ancho)
    # en pares de variable-valor hacia abajo (largo) para dibujarlos de golpe
    factores = ["Smoking", "Alcohol_Use", "Obesity_Level"]
    df_melted = df.melt(
        id_vars=["Rango_Edad"],
        value_vars=factores,
        var_name="Factor_Riesgo",
        value_name="Magnitud_Nivel",
    )

    fig_fact, ax_fact = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_melted,
        x="Factor_Riesgo",
        y="Magnitud_Nivel",
        hue="Rango_Edad",
        palette="Set3",
        ax=ax_fact,
    )
    ax_fact.set_title("Dispersión de los hábitos perjudiciales categorizado por edad")
    st.pyplot(fig_fact)

    st.divider()

    # --- 4. TABLA RESUMEN GENERAL ---
    st.subheader("Resumen ejecutivo (medias comparativas)")
    # Muestra una tabla comparando la media de cada riesgo entre el mundo real vs la simulación
    resumen = df.groupby("Origen")[factores + ["Target_Severity_Score"]].mean()
    st.table(resumen)
