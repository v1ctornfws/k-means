import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings

# Suprimir la advertencia que a veces aparece con n_init='auto' en versiones de sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

# ====================================================================
# FUNCIÓN 1: Preparación y Escalamiento de Datos
# ====================================================================

def preparar_y_escalar_datos(df: pd.DataFrame, columnas: list):
    """
    Selecciona las columnas para el clustering, escala los datos usando MinMaxScaler 
    y devuelve los datos originales, escalados y el objeto escalador.
    """
    
    # 1. Seleccionar columnas
    X = df[columnas]
    
    # 2. Instanciar y ajustar el escalador
    escalador = MinMaxScaler()
    X_escalado = escalador.fit_transform(X)
    
    # 3. Convertir a DataFrame para el clustering
    df_escalado = pd.DataFrame(X_escalado, columns=columnas)
    
    return X, df_escalado, escalador


# ====================================================================
# FUNCIÓN 2: Método del Codo (Inercia)
# ====================================================================

def calcular_inercia(df_escalado: pd.DataFrame, max_k: int = 10):
    """
    Calcula la inercia (WCSS) para un rango de K para aplicar el método del codo.
    """
    
    inercia = []
    K_rango = range(1, max_k + 1)
    
    for k in K_rango:
        # Usamos n_init='auto' para el cálculo
        modelo = KMeans(n_clusters=k, random_state=42, n_init='auto')
        modelo.fit(df_escalado)
        inercia.append(modelo.inertia_)
        
    return K_rango, inercia


# ====================================================================
# FUNCIÓN 3: Aplicar K-means Final
# ====================================================================

def aplicar_kmeans(df_escalado: pd.DataFrame, k_optimo: int):
    """
    Entrena el modelo final de K-means con el k óptimo elegido.
    """
    
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init='auto')
    kmeans_final.fit(df_escalado)
    
    # 1. Etiquetas de cluster para cada punto de dato
    etiquetas = kmeans_final.labels_
    
    # 2. Coordenadas de los centroides escalados
    centroides_escalados = kmeans_final.cluster_centers_
    
    return etiquetas, centroides_escalados