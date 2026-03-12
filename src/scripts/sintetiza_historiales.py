########################################################################################
# Verion 1 - sintetiza historiales ficticios de pacientes
########################################################################################

import os

import pandas as pd
import numpy as np

# Configurar semilla para que siempre salgan los mismos datos si lo vuelves a ejecutar
np.random.seed(42)

# Número de pacientes a generar
n_pacientes = 5000

# -------------------------------------------------------------------------------------------
# 1. Crear el Target (El diagnóstico): 50% sanos (0) y 50% enfermos (1)
diagnostico = np.random.choice([0, 1], size=n_pacientes)

# -------------------------------------------------------------------------------------------
# 2. Generar Edad entre 50 y 69 años
edad = np.random.normal(loc=60, scale=8, size=n_pacientes)
edad = np.clip(edad, 50, 69).astype(int)

# -------------------------------------------------------------------------------------------
# 3. Generar Género
genero = np.random.choice(["Male", "Female"], size=n_pacientes)

# -------------------------------------------------------------------------------------------
# 4. Variables de Riesgo
# Los pacientes enfermos (target=1) tendrán promedios más altos, pero con variación

# Fuma
fumador = np.where(
    diagnostico == 1,
    np.random.normal(loc=5.0, scale=3.0, size=n_pacientes),
    np.random.normal(loc=3.0, scale=2.5, size=n_pacientes),
)
fumador = np.clip(fumador, 0, 10).round(1)

# Consumo de alcohol
consume_alcohol = np.where(
    diagnostico == 1,
    np.random.normal(loc=5.5, scale=2.5, size=n_pacientes),
    np.random.normal(loc=3.5, scale=2.0, size=n_pacientes),
)
consume_alcohol = np.clip(consume_alcohol, 0, 10).round(1)

# Dieta rica en grasas animales (carnes rojas), pobre en fibra, fruta y verduras
# Si está enfermo (1), tiene un 70% de probabilidad de tener mala dieta. Si está sano (0), solo un 30%.
prob_dieta = np.where(diagnostico == 1, 0.70, 0.30)
Dieta_rica_en_grasas_animales = np.random.binomial(n=1, p=prob_dieta)

# Sedentarismo
# Si está enfermo, 65% de probabilidad de ser sedentario. Si está sano, 40%.
prob_sedentarismo = np.where(diagnostico == 1, 0.65, 0.40)
sedentarismo = np.random.binomial(n=1, p=prob_sedentarismo)

# Obesidad
nivel_obesidad = np.where(
    diagnostico == 1,
    np.random.normal(loc=6.5, scale=2.0, size=n_pacientes),
    np.random.normal(loc=4.0, scale=1.5, size=n_pacientes),
)
nivel_obesidad = np.clip(nivel_obesidad, 0, 10).round(1)

# Diabetes tipo 2
# La diabetes aumenta el riesgo, pero no es tan común.
# Si está enfermo, 25% de probabilidad de tener diabetes. Si está sano, 10%.
prob_diabetes = np.where(diagnostico == 1, 0.25, 0.10)
diabetes_tipo_2 = np.random.binomial(n=1, p=prob_diabetes)

# -------------------------------------------------------------------------
# HISTORIAL MÉDICO: FACTORES HEREDITARIOS E INFLAMATORIOS (Basado en el PDF asociacion ESP contra el cancer)
# -------------------------------------------------------------------------

# 1. Agrupación familiar (Hasta un 25% de los pacientes con cáncer tienen un familiar afecto)
# Asumimos un 25% en enfermos y un 10% en la población sana (por pura estadística poblacional)
prob_familiar = np.where(diagnostico == 1, 0.25, 0.10)
antecedentes_familiares = np.random.binomial(n=1, p=prob_familiar)

# 2. Componente hereditario (< 10% de todos los casos)
# Le daremos un 8% a los enfermos y un 2% a los sanos.
prob_hereditario = np.where(diagnostico == 1, 0.08, 0.02)
componente_hereditario = np.random.binomial(n=1, p=prob_hereditario)

# 3. Síndromes predisponentes (< 5% de todos los casos)
# Le daremos un 4% a los enfermos y un 0.5% a los sanos.
prob_sindromes = np.where(diagnostico == 1, 0.04, 0.005)
sindromes_predisponentes = np.random.binomial(n=1, p=prob_sindromes)

# 4. Enfermedades Inflamatorias Intestinales (EII: Crohn, Colitis Ulcerosa)
# El texto dice: < 1% de todos los cánceres.
# Le daremos un 0.9% a los enfermos y un 0.4% a los sanos.
prob_eii = np.where(diagnostico == 1, 0.009, 0.004)
enfermedad_inflamatoria_intestinal = np.random.binomial(n=1, p=prob_eii)

# -------------------------------------------------------------------------------------------
# 5. Pruebas Médicas (La clave del diagnóstico temprano para tu simulador)

# Sangre Oculta en Heces (FOBT - Prueba General):
# Si está enfermo, 85% probabilidad de positivo. Si está sano, solo 5% (falsos positivos comunes)
fobt_prob = np.where(diagnostico == 1, 0.85, 0.05)
fobt_result = np.random.binomial(n=1, p=fobt_prob)
fobt_result = ["Positive" if res == 1 else "Negative" for res in fobt_result]

# Marcador Tumoral CEA (Prueba Tumoral):
# Lo normal es < 5 ng/mL.
cea_level = np.where(
    diagnostico == 1,
    np.random.normal(
        loc=8.5, scale=4.0, size=n_pacientes
    ),  # Enfermos: valores más altos
    np.random.normal(loc=2.5, scale=1.0, size=n_pacientes),
)  # Sanos: valores bajos
cea_level = np.clip(cea_level, 0.1, 50.0).round(2)

# -------------------------------------------------------------------------------------------
# 6. Crear el DataFrame
df_simulador = pd.DataFrame(
    {
        "Paciente_ID": [f"PT{str(i).zfill(5)}" for i in range(n_pacientes)],
        "Edad": edad,
        "Genero": genero,
        "Fumador": fumador,
        "Consume_Alcohol": consume_alcohol,
        "Dieta_rica_en_grasas_animales": Dieta_rica_en_grasas_animales,
        "Sedentarismo": sedentarismo,
        "Obesidad": nivel_obesidad,
        "Diabetes_tipo_2": diabetes_tipo_2,
        "Antecedentes_Familiares": antecedentes_familiares,
        "Componente_Hereditario": componente_hereditario,
        "Sindromes_Predisponentes": sindromes_predisponentes,
        "Enfermedad_Inflamatoria_Intestinal": enfermedad_inflamatoria_intestinal,
        "FOBT_Resultado (Sangre en heces)": fobt_result,  # Prueba General
        "CEA_Level_ng_mL (Marcador Tumoral)": cea_level,  # Prueba Tumoral
        "Diagnostico": diagnostico,  # 0 = Sano, 1 = Riesgo de Cáncer/Pólipos
    }
)

# Guardar en CSV
df_simulador.to_csv(
    "src/data/raw/historial_pacientes/historiales_sinteticos/pacientes_simulador_colon.csv",
    index=False,
)

def sintetizar_historiales(df_global, output_dir):

    df_global = df_global.copy()

    df_global['City'] = 'Barcelona'
    df_global['Country'] = 'Spain'

    if 'Target_Severity_Score' in df_global.columns:
        df_global['diagnostico'] = (df_global['Target_Severity_Score'] > 4).astype(int)
    else:
        # Si no hay score, asumimos una probabilidad base para no romper el código
        df_global['diagnostico'] = np.random.binomial(1, 0.5, size=len(df_global))

    # 2. Ahora sí, las probabilidades funcionarán porque 'diagnostico' tiene datos
    n = len(df_global)
    # sintetizar datos para las nuevas columnas
    df_global['Dieta_rica_en_grasas_animales'] = np.random.binomial(n=1, p=0.3, size=len(df_global))

    # Si está enfermo, 65% de probabilidad de ser sedentario. Si está sano, 40%.
    prob_sedentarismo = np.where(df_global['diagnostico'] == 1, 0.65, 0.40)
    df_global['Sedentarismo'] = np.random.binomial(n=1, p=prob_sedentarismo, size=len(df_global))

    # La diabetes aumenta el riesgo, pero no es tan común.
    # Si está enfermo, 25% de probabilidad de tener diabetes. Si está sano, 10%.
    prob_diabetes = np.where(df_global['diagnostico'] == 1, 0.25, 0.10)
    df_global['Diabetes_tipo_2'] = np.random.binomial(n=1, p=prob_diabetes, size=len(df_global))

    prob_familiar = np.where(df_global['diagnostico'] == 1, 0.25, 0.10)
    df_global['Antecedentes_Familiares'] = np.random.binomial(n=1, p=prob_familiar, size=len(df_global))

    # 2. Componente hereditario (< 10% de todos los casos)
    # Le daremos un 8% a los enfermos y un 2% a los sanos.
    prob_hereditario = np.where(df_global['diagnostico'] == 1, 0.08, 0.02)
    df_global['Componente_Hereditario'] = np.random.binomial(n=1, p=prob_hereditario, size=len(df_global))

    # 3. Síndromes predisponentes (< 5% de todos los casos)
    # Le daremos un 4% a los enfermos y un 0.5% a los sanos.
    prob_sindromes = np.where(df_global['diagnostico'] == 1, 0.04, 0.005)
    df_global['Sindromes_Predisponentes'] = np.random.binomial(n=1, p=prob_sindromes, size=len(df_global))

    # 4. Enfermedades Inflamatorias Intestinales (EII: Crohn, Colitis Ulcerosa)
    # El texto dice: < 1% de todos los cánceres.
    # Le daremos un 0.9% a los enfermos y un 0.4% a los sanos.
    prob_eii = np.where(df_global['diagnostico'] == 1, 0.009, 0.004)
    df_global['Enfermedad_Inflamatoria_Intestinal'] = np.random.binomial(n=1, p=prob_eii, size=len(df_global))

    # -------------------------------------------------------------------------------------------
    # 5. Pruebas Médicas (La clave del diagnóstico temprano para tu simulador)

    # Sangre Oculta en Heces (FOBT - Prueba General):
    # Si está enfermo, 85% probabilidad de positivo. Si está sano, solo 5% (falsos positivos comunes)
    fobt_prob = np.where(df_global['diagnostico'] == 1, 0.85, 0.05)
    df_global['FOBT_Resultado_n'] = np.random.binomial(n=1, p=fobt_prob, size=len(df_global))
    df_global['FOBT_Resultado (Sangre en heces)'] = df_global['FOBT_Resultado_n'].map({1: 'Positive', 0: 'Negative'})

    # Marcador Tumoral CEA (Prueba Tumoral):
    # Lo normal es < 5 ng/mL.
    cea_level = np.where(
        df_global['diagnostico'] == 1,
        np.random.normal(
            loc=8.5, scale=4.0, size=len(df_global)
        ),  # Enfermos: valores más altos
        np.random.normal(loc=2.5, scale=1.0, size=len(df_global)),
    )  # Sanos: valores bajos
    cea_level = np.clip(cea_level, 0.1, 50.0).round(2)
    df_global['CEA_Level_ng_mL (Marcador Tumoral)'] = cea_level

    df_global['FOBT_Resultado_n'] = df_global['FOBT_Resultado_n'].astype(int)

    output_path = os.path.join(output_dir, 'datos_combinados_global_extendido_3.csv')
    df_global.to_csv(output_path, index=False)

    return df_global

def sintetizar_datos_kaggle(df_k, output_dir):
    df_k = df_k.copy()

    # 1. Definir columnas base
    df_k['City'] = 'Barcelona'
    df_k['Country'] = 'Spain'
    
    # Usamos 'Mortality' como indicador de enfermedad grave/diagnóstico para la síntesis
    # Si 'Mortality' no es la que quieres usar, cámbiala por 'Cancer_Stage' o la que prefieras
    condicion = df_k['Mortality'] 

    # 2. Generar datos sintéticos basados en condiciones
    n = len(df_k)

    # Dieta (Probabilidad fija)
    df_k['Dieta_rica_en_grasas_animales'] = np.random.binomial(n=1, p=0.3, size=n)

    # Sedentarismo: Si Mortality=1 (enfermo), 65% prob. Si 0, 40% prob.
    prob_sed = np.where(condicion == 1, 0.65, 0.40)
    df_k['Sedentarismo'] = np.random.binomial(n=1, p=prob_sed)

    # Diabetes tipo 2
    prob_diab = np.where(condicion == 1, 0.25, 0.10)
    df_k['Diabetes_tipo_2'] = np.random.binomial(n=1, p=prob_diab)

    # Antecedentes Familiares
    prob_fam = np.where(condicion == 1, 0.25, 0.10)
    df_k['Antecedentes_Familiares'] = np.random.binomial(n=1, p=prob_fam)

    # Componente hereditario
    prob_herd = np.where(condicion == 1, 0.08, 0.02)
    df_k['Componente_Hereditario'] = np.random.binomial(n=1, p=prob_herd)

    # Síndromes predisponentes
    prob_sind = np.where(condicion == 1, 0.04, 0.005)
    df_k['Sindromes_Predisponentes'] = np.random.binomial(n=1, p=prob_sind)

    # Enfermedad Inflamatoria Intestina
    prob_eii = np.where(condicion == 1, 0.009, 0.004)
    df_k['Enfermedad_Inflamatoria_Intestinal'] = np.random.binomial(n=1, p=prob_eii)

    # FOBT (Sangre en heces) - Probabilidades corregidas (0.85, no 85)
    prob_fobt = np.where(condicion == 1, 0.85, 0.05)
    df_k['FOBT_Resultado_n'] = np.random.binomial(n=1, p=prob_fobt)
    df_k['FOBT_Resultado (Sangre en heces)'] = df_k['FOBT_Resultado_n'].map({1: 'Positive', 0: 'Negative'})

    # 3. Marcador Tumoral CEA (Distribución Normal)
    # Generamos valores para ambos casos y elegimos con np.where
    cea_enfermo = np.random.normal(loc=8.5, scale=4.0, size=n)
    cea_sano = np.random.normal(loc=2.5, scale=1.0, size=n)
    
    df_k['CEA_Level_ng_mL (Marcador Tumoral)'] = np.where(condicion == 1, cea_enfermo, cea_sano)
    df_k['CEA_Level_ng_mL (Marcador Tumoral)'] = df_k['CEA_Level_ng_mL (Marcador Tumoral)'].clip(0.1, 50.0).round(2)

    # Asegurar tipos
    df_k['FOBT_Resultado_n'] = df_k['FOBT_Resultado_n'].astype(int)

    # 4. Guardado
    output_path = os.path.join(output_dir, 'datos_finales_Kaggle.csv')
    df_k.to_csv(output_path, index=False)

    return df_k