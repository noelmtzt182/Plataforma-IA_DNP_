import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error
)

st.set_page_config(page_title="Modelo ML", layout="wide")
st.title("Modelo de Clasificación y Regresión")

# ============================
# 1. Cargar datos
# ============================
csv_path = "mercado_cereales_5000_con_ventas.csv"

try:
    df = pd.read_csv(csv_path)
    st.success(f"Archivo cargado correctamente: {csv_path}")
except Exception as e:
    st.error(f"No se pudo cargar el archivo CSV: {e}")
    st.stop()

st.write("Vista previa del dataset")
st.dataframe(df.head())

# ============================
# 2. Limpieza básica
# ============================
for col in ["marca", "categoria", "canal", "estacionalidad", "comentario"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

numeric_cols = [
    "precio", "costo", "margen", "margen_pct",
    "competencia", "demanda", "tendencia",
    "rating_conexion", "sentiment_score",
    "conexion_score", "conexion_alta",
    "score_latente", "exito", "ventas_unidades"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    "marca", "canal", "precio", "competencia", "demanda",
    "tendencia", "margen_pct", "conexion_score",
    "rating_conexion", "sentiment_score", "exito", "ventas_unidades"
])

df["exito"] = df["exito"].astype(int)

st.write("Dimensiones del dataset limpio:", df.shape)

# ============================
# 3. Definir variables
# ============================
features = [
    "precio", "competencia", "demanda", "tendencia", "margen_pct",
    "conexion_score", "rating_conexion", "sentiment_score",
    "marca", "canal"
]

X = df[features].copy()
y_class = df["exito"].copy()
y_reg = df["ventas_unidades"].copy()

num_cols = [
    "precio", "competencia", "demanda", "tendencia",
    "margen_pct", "conexion_score", "rating_conexion", "sentiment_score"
]
cat_cols = ["marca", "canal"]

# ============================
# 4. Preprocesamiento
# ============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ============================
# 5. Modelo de clasificación
# ============================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

clf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

clf_model.fit(X_train_c, y_train_c)

y_pred_c = clf_model.predict(X_test_c)
y_prob_c = clf_model.predict_proba(X_test_c)[:, 1]

accuracy = accuracy_score(y_test_c, y_pred_c)
auc = roc_auc_score(y_test_c, y_prob_c)
cm = confusion_matrix(y_test_c, y_pred_c)

st.subheader("Resultados del modelo de clasificación")
c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("ROC-AUC", f"{auc:.4f}")

st.write("Matriz de confusión")
st.dataframe(pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]))

# ============================
# 6. Modelo de regresión
# ============================
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)
mae = mean_absolute_error(y_test_r, y_pred_r)

st.subheader("Resultados del modelo de regresión")
st.metric("MAE", f"{mae:.2f}")

# ============================
# 7. Ejemplo de predicción
# ============================
nuevo_producto = pd.DataFrame([{
    "precio": 52.0,
    "competencia": 7,
    "demanda": 78,
    "tendencia": 82,
    "margen_pct": 35,
    "conexion_score": 74,
    "rating_conexion": 8,
    "sentiment_score": 1,
    "marca": "nueva",
    "canal": "retail"
}])

prob_exito = clf_model.predict_proba(nuevo_producto)[0][1]
ventas_estimadas = reg_model.predict(nuevo_producto)[0]

st.subheader("Predicción para nuevo producto")
p1, p2 = st.columns(2)
p1.metric("Probabilidad de éxito", f"{prob_exito*100:.2f}%")
p2.metric("Ventas estimadas", f"{ventas_estimadas:.0f} unidades")
