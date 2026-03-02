import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estética premium
plt.style.use('dark_background')
palette = sns.color_palette("viridis", as_cmap=True)

# Rutas de archivos
data_path = r'c:\Users\IVONNE ENRIQUEZ\Documents\Taller ML\DATA\earthquakes_colombia.csv'
img_dir = r'c:\Users\IVONNE ENRIQUEZ\Documents\Taller ML\IMAGENES'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

def perform_eda():
    # 1. Cargar datos
    print("Cargando datos...")
    df = pd.read_csv(data_path)
    
    # 2. Análisis de valores nulos
    print("\n--- Análisis de Valores Nulos ---")
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({'Counts': null_counts, 'Percentage': null_percentages})
    null_df = null_df[null_df['Counts'] > 0].sort_values(by='Counts', ascending=False)
    print(null_df)
    
    # 3. Distribuciones de Magnitud y Profundidad
    print("\nGenerando distribuciones...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df['mag'], bins=30, kde=True, ax=axes[0], color='#00d1b2', edgecolor='white')
    axes[0].set_title('Distribución de Magnitudes', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Magnitud')
    axes[0].set_ylabel('Frecuencia')
    
    sns.histplot(df['depth'], bins=30, kde=True, ax=axes[1], color='#ff3860', edgecolor='white')
    axes[1].set_title('Distribución de Profundidades', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Profundidad (km)')
    axes[1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'distributions.png'), dpi=300)
    plt.close()
    
    # 4. Correlaciones
    print("Calculando correlaciones...")
    # Solo variables numéricas
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 5. Mapas Scatter
    print("Generando mapas scatter...")
    
    # Mapa por Profundidad
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df['depth'], 
                         cmap='magma', alpha=0.6, s=df['mag']*10)
    plt.colorbar(scatter, label='Profundidad (km)')
    plt.title('Mapa Sísmico: Latitud vs Longitud (Color por Profundidad)', fontsize=14)
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(img_dir, 'map_depth.png'), dpi=300)
    plt.close()
    
    # Mapa por Magnitud
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(df['longitude'], df['latitude'], c=df['mag'], 
                         cmap='viridis', alpha=0.6, s=df['mag']*10)
    plt.colorbar(scatter, label='Magnitud')
    plt.title('Mapa Sísmico: Latitud vs Longitud (Color por Magnitud)', fontsize=14)
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(img_dir, 'map_magnitude.png'), dpi=300)
    plt.close()
    
    print("\nEDA completado. Imágenes guardadas en la carpeta IMAGENES.")

if __name__ == "__main__":
    perform_eda()
