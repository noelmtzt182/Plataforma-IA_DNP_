import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Cargar dataset
df = pd.read_csv("mercado_cereales_5000_con_ventas.csv")

# Selección de variables
features = [
    "precio", "competencia", "demanda", "tendencia", "margen_pct",
    "conexion_score", "rating_conexion", "sentiment_score",
    "marca", "canal"
]

X = df[features]
y_class = df["exito"]
y_reg = df["ventas_unidades"]

# Variables numéricas y categóricas
num_cols = [
    "precio", "competencia", "demanda", "tendencia",
    "margen_pct", "conexion_score", "rating_conexion", "sentiment_score"
]
cat_cols = ["marca", "canal"]

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# =========================
# Modelo de clasificación
# =========================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

clf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

clf_model.fit(X_train_c, y_train_c)

y_pred_c = clf_model.predict(X_test_c)
y_prob_c = clf_model.predict_proba(X_test_c)[:, 1]

accuracy = accuracy_score(y_test_c, y_pred_c)
auc = roc_auc_score(y_test_c, y_prob_c)

print("Accuracy:", accuracy)
print("ROC-AUC:", auc)

# =========================
# Modelo de regresión
# =========================
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)
mae = mean_absolute_error(y_test_r, y_pred_r)

print("MAE:", mae)
