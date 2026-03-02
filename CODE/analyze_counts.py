import pandas as pd

def analyze_dataset():
    path = 'DATA/earthquakes_colombia.csv'
    try:
        df = pd.read_csv(path)
        total_records = len(df)
        
        # Count records that mention "Colombia" in the 'place' column
        colombia_records = df[df['place'].str.contains('Colombia', na=False, case=False)].shape[0]
        
        print(f"Total de registros en el CSV: {total_records}")
        print(f"Registros identificados específicamente en Colombia: {colombia_records}")
        
    except Exception as e:
        print(f"Error al analizar el archivo: {e}")

if __name__ == "__main__":
    analyze_dataset()
