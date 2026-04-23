import pandas as pd
import os

def unify_csvs():
    # Obtener rutas de los archivos directamente para el script puntual
    base_dir = r"c:\Users\User\Desktop\Programacion\Proyecto 2 - Cancer colon\src\data\clean"
    ready_dir = r"c:\Users\User\Desktop\Programacion\Proyecto 2 - Cancer colon\src\data\ready"
    
    csv_nuevos_5000 = os.path.join(base_dir, "nuevos_pacientes_5000.csv")
    csv_risk_final = os.path.join(base_dir, "cancer_risk_final.csv")
    
    # Crear carpeta ready si no existe
    os.makedirs(ready_dir, exist_ok=True)
    
    print(f"Cargando {csv_nuevos_5000}...")
    df_nuevos = pd.read_csv(csv_nuevos_5000)
    print(f"Columnas en nuevos: {list(df_nuevos.columns)}")
    
    print(f"Cargando {csv_risk_final}...")
    df_risk = pd.read_csv(csv_risk_final)
    print(f"Columnas en risk_final: {list(df_risk.columns)}")
    
    print("Realizando Join por Patient_ID...")
    # Verificamos si existe Patient_ID en ambos
    if "Patient_ID" in df_nuevos.columns and "Patient_ID" in df_risk.columns:
        df_master = pd.merge(df_nuevos, df_risk, on="Patient_ID", how="inner")
        
        # Eliminar las posibles columnas duplicadas si hubiese (como Gender o Age)
        # En este caso, no deberían superponerse excepto Patient_ID, pero por si acaso.
        
        output_path = os.path.join(ready_dir, "pacientes_master.csv")
        df_master.to_csv(output_path, index=False)
        print(f"Merge exitoso! Guardado en {output_path} con {len(df_master)} filas.")
        print(f"Columnas resultantes ({len(df_master.columns)}): {list(df_master.columns)}")
    else:
        print("Error: Patient_ID no encontrado en alguno de los dataframes.")

if __name__ == "__main__":
    unify_csvs()
