import pandas as pd
import numpy as np
import os
from datetime import datetime

def advanced_engineering():
    input_path = 'DATA/earthquakes_filtered.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} no existe.")
        return

    df = pd.read_csv(input_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    print("Iniciando Ingeniería de Características Avanzada...")

    # 1. Energía Liberada (Julios) - Ley de Gutenberg-Richter
    # log10(E) = 4.8 + 1.5 * M  => E = 10^(4.8 + 1.5 * M)
    df['energy_joules'] = 10**(4.8 + 1.5 * df['mag'])
    print("- Energía liberada calculada.")

    # 2. Variables Temporales
    # Días desde el primer evento
    start_date = df['time'].min()
    df['days_since_start'] = (df['time'] - start_date).dt.days
    
    # Frecuencia temporal (Sismos en los últimos 30 días - Ventana móvil)
    # Nota: Requiere un poco más de cómputo, usamos rolling
    df.set_index('time', inplace=True)
    df['rolling_30d_count'] = df.rolling('30D').count()['id']
    df.reset_index(inplace=True)
    print("- Variables temporales y densidad calculadas.")

    # 3. Variables Espaciales
    # Distancia al geocentro (simplificado: asumiendo esfera)
    # Combinando coordenadas y profundidad
    R_earth = 6371.0
    # Convertir a radianes
    lat_rad = np.radians(df['latitude'])
    lon_rad = np.radians(df['longitude'])
    
    # Coordenadas Cartesianas 3D considerando profundidad (depth)
    r = R_earth - df['depth']
    df['pos_x'] = r * np.cos(lat_rad) * np.cos(lon_rad)
    df['pos_y'] = r * np.cos(lat_rad) * np.sin(lon_rad)
    df['pos_z'] = r * np.sin(lat_rad)
    print("- Coordenadas cartesianas 3D (incorporando profundidad) generadas.")

    # 4. Distancia a Ciudades Principales (Riesgo)
    cities = {
        'Bogota': (4.7110, -74.0721),
        'Medellin': (6.2442, -75.5812),
        'Cali': (3.4516, -76.5320),
        'Bucaramanga': (7.1193, -73.1227),
        'Pasto': (1.2136, -77.2811)
    }

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0 # Radio Tierra
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    for city, coords in cities.items():
        df[f'dist_to_{city}'] = haversine(df['latitude'], df['longitude'], coords[0], coords[1])
    
    df['min_dist_to_major_city'] = df[[f'dist_to_{city}' for city in cities.keys()]].min(axis=1)
    print("- Distancias a centros poblados calculadas.")

    # Guardar dataset enriquecido
    output_path = 'DATA/earthquakes_advanced.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset enriquecido guardado en: {output_path}")
    print(f"Nuevas columnas disponibles: {list(df.columns[-10:])}")

if __name__ == "__main__":
    advanced_engineering()
