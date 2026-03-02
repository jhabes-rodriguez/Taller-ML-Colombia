import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def prepare_data():
    input_path = 'DATA/earthquakes_colombia.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} no existe.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Filtrar solo los registros que corresponden a Colombia (basado en la columna 'place')
    # Esto reduce el ruido de eventos lejanos capturados por el área de búsqueda
    df_colombia = df[df['place'].str.contains('Colombia', na=False, case=False)].copy()
    
    print(f"Registros originales: {len(df)}")
    print(f"Registros filtrados (Colombia): {len(df_colombia)}")

    # 2. Selección de variables instrumentales para el clustering
    features = ['latitude', 'longitude', 'depth', 'mag']
    X = df_colombia[features]

    # 3. Escalamiento de datos (StandardScaler)
    # Esto es crucial para que la profundidad (kms) no domine sobre lat/lon (grados)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Crear un nuevo dataframe con los datos escalados
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Mantener los índices originales para poder unir con los datos originales después si es necesario
    df_scaled.index = df_colombia.index

    # 4. Guardar los datos preparados
    os.makedirs("DATA", exist_ok=True)
    scaler_output = 'DATA/normalized_data.csv'
    df_scaled.to_csv(scaler_output, index=False)
    
    # Guardar también el subset filtrado de Colombia sin escalar para referencia
    subset_output = 'DATA/earthquakes_filtered.csv'
    df_colombia.to_csv(subset_output, index=False)
    
    print(f"Datos escalados guardados en: {scaler_output}")
    print(f"Datos filtrados guardados en: {subset_output}")
    
    # Mostrar las primeras filas de los datos escalados
    print("\nPrimeras filas de los datos escalados:")
    print(df_scaled.head())

if __name__ == "__main__":
    prepare_data()
