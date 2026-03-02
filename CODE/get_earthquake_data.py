import pandas as pd
import os

def get_data():
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        "?format=csv&starttime=2010-01-01&endtime=2026-02-20"
        "&minlatitude=-4.5&maxlatitude=13.5"
        "&minlongitude=-82&maxlongitude=-66.5"
        "&minmagnitude=1.5&orderby=time&limit=20000"
    )

    print(f"Descargando datos desde USGS...")
    try:
        df = pd.read_csv(url)
        print(f"Registros encontrados: {len(df)}")

        # Ensure DATA directory exists
        os.makedirs("DATA", exist_ok=True)
        
        output_path = os.path.join("DATA", "earthquakes_colombia.csv")
        df.to_csv(output_path, index=False)
        print(f"Base de datos guardada en: {output_path}")
        return True
    except Exception as e:
        print(f"Error al descargar los datos: {e}")
        return False

if __name__ == "__main__":
    get_data()
