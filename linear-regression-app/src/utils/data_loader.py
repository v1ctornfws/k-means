import pandas as pd


def load_data(file):
    """
    Carga los datos desde un archivo CSV
    """
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {str(e)}")
