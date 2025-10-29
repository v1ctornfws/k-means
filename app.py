import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importar la lógica del modelo del archivo local
from model_utils import (
    preparar_y_escalar_datos, 
    calcular_inercia, 
    aplicar_kmeans
)

# Definición de las columnas (ajusta si tus nombres son diferentes)
COLUMNAS_CLUSTERING = ['Ingresos Anuales', 'Puntuación de Gasto']


st.set_page_config(layout="wide")
st.title("🛍️ Segmentación de Clientes con K-means")
st.markdown("---")

# ====================================================================
# 1. Carga de Datos y Preprocesamiento
# ====================================================================

# Cargar los datos. st.cache_data asegura que solo se ejecute la primera vez.
@st.cache_data 
def cargar_y_preprocesar():
    try:
        # Asume que 'clientes.csv' está en la carpeta 'data/'
        df = pd.read_csv("data/clientes.csv") 
    except FileNotFoundError:
        st.error("Error: Archivo 'data/clientes.csv' no encontrado. Asegúrate de tener la estructura: data/clientes.csv")
        return None, None, None, None
        
    # Preparar y escalar datos
    df_copia = df.copy()
    X_original, df_escalado, escalador = preparar_y_escalar_datos(df_copia, COLUMNAS_CLUSTERING)
    
    return df_copia, X_original, df_escalado, escalador

df, X_original, df_escalado, escalador = cargar_y_preprocesar()

if df is not None:
    
    st.header("1. Datos Originales")
    st.markdown(f"Usando las columnas **{COLUMNAS_CLUSTERING[0]}** y **{COLUMNAS_CLUSTERING[1]}**.")
    st.dataframe(df.head(10))
    st.markdown("---")
    
    # ====================================================================
    # 2. Método del Codo Interactivo
    # ====================================================================
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("2. Método del Codo")
        st.markdown("Elige el número de clusters (K) donde la curva se dobla (el 'codo').")
        
        # Interfaz para que el usuario elija K
        k_optimo = st.slider(
            "Elige K:", 
            min_value=2, max_value=10, value=5, 
            key="k_slider"
        )
        
    with col2:
        # Cálculo y Gráfica del Codo
        K_rango, inercia = calcular_inercia(df_escalado)
        
        fig_codo, ax_codo = plt.subplots(figsize=(8, 4))
        ax_codo.plot(K_rango, inercia, marker='o', linestyle='--', color='blue')
        ax_codo.set_title('Método del Codo')
        ax_codo.set_xlabel('Número de Clusters (k)')
        ax_codo.set_ylabel('Inercia (WCSS)')
        ax_codo.axvline(x=k_optimo, color='r', linestyle='--', label=f'K={k_optimo} Seleccionado')
        ax_codo.legend()
        ax_codo.grid(True)
        st.pyplot(fig_codo, use_container_width=True)
        
    st.markdown("---")
    
    # ====================================================================
    # 3. Aplicación del Modelo Final y Visualización
    # ====================================================================
    
    st.header(f"3. Resultados de Segmentación Final (K = {k_optimo})")
    
    # Llamar a la función de utilidades para aplicar K-means
    etiquetas, centroides_escalados = aplicar_kmeans(df_escalado, k_optimo)
    
    # Añadir las etiquetas al DataFrame original para el análisis
    df['Cluster'] = etiquetas
    
    # Transformación inversa de los centroides a la escala original
    centroides_originales = escalador.inverse_transform(centroides_escalados)
    
    
    # A. Análisis de Perfiles de Cluster
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("A. Perfiles Promedio de los Clusters")
        st.markdown(f"Características medias de los {k_optimo} grupos:")
        perfiles = df.groupby('Cluster')[COLUMNAS_CLUSTERING].mean()
        st.dataframe(perfiles.style.format("{:.2f}"))
        
    # B. Gráfica de Segmentación
    with col4:
        st.subheader("B. Distribución de Clusters")
        
        fig_final, ax_final = plt.subplots(figsize=(10, 6))
        
        # Scatter plot de los puntos de datos (clientes)
        scatter = ax_final.scatter(df[COLUMNAS_CLUSTERING[0]], df[COLUMNAS_CLUSTERING[1]], 
                                   c=df['Cluster'], cmap='viridis', s=80, alpha=0.7)
        
        # Scatter plot de los centroides (marcadores 'X')
        ax_final.scatter(centroides_originales[:, 0], centroides_originales[:, 1], 
                         marker='X', s=300, c='red', edgecolor='black', 
                         label='Centroides')
        
        ax_final.set_title(f'Segmentación de Clientes con K={k_optimo}')
        ax_final.set_xlabel(COLUMNAS_CLUSTERING[0])
        ax_final.set_ylabel(COLUMNAS_CLUSTERING[1])
        ax_final.legend()
        ax_final.grid(True)
        
        st.pyplot(fig_final, use_container_width=True)