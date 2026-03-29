import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Debug App", layout="wide")
st.title("Debug de arranque ✅")

st.write("Hola, la app sí está corriendo.")

csv_path = "mercado_cereales_5000_con_ventas.csv"
st.write("Archivo esperado:", csv_path)
st.write("Existe?", Path(csv_path).exists())

if Path(csv_path).exists():
    df = pd.read_csv(csv_path)
    st.write("CSV cargado correctamente")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
else:
    st.error("No se encontró el CSV")
