
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar y entrenar el modelo si no existe
model_path = "modelo.pkl"

@st.cache_resource
def cargar_o_entrenar_modelo():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        df = pd.read_csv("Liga1Proce.csv")
        X = df[["Goles Local", "Goles Visitante"]]
        y = df["Resultado Codificado"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        return model

# Cargar dataset para uso en la app
df = pd.read_csv("Liga1Proce.csv")
model = cargar_o_entrenar_modelo()

# Título y descripción
st.title("📊 Predicción de Resultados - Liga 1 Peruana 2025")
st.subheader("Sistema de predicción usando IA y Aprendizaje Estadístico")
st.markdown("---")

# Selección de equipos
st.markdown("### Seleccione los equipos del partido")
local = st.selectbox("🔴 Equipo Local", sorted(df["Local"].unique()))
visitante = st.selectbox("🔵 Equipo Visitante", sorted(df["Visitante"].unique()))

# Cálculo de promedios
avg_local = df[df["Local"] == local]["Goles Local"].mean()
avg_visit = df[df["Visitante"] == visitante]["Goles Visitante"].mean()

st.markdown("### Estadísticas Promedio")
col1, col2 = st.columns(2)
col1.metric(f"Goles promedio de {local} (local)", f"{avg_local:.2f}")
col2.metric(f"Goles promedio de {visitante} (visitante)", f"{avg_visit:.2f}")

st.markdown("### Ajuste los goles esperados")
goles_local = st.number_input("⚽ Goles esperados del equipo Local", min_value=0.0, max_value=10.0, step=0.5, value=float(f"{avg_local:.2f}"))
goles_visitante = st.number_input("⚽ Goles esperados del equipo Visitante", min_value=0.0, max_value=10.0, step=0.5, value=float(f"{avg_visit:.2f}"))

# Botón para predecir
if st.button("🔮 Predecir Resultado"):
    entrada = np.array([[goles_local, goles_visitante]])
    prediccion = model.predict(entrada)[0]

    if prediccion == 1:
        resultado = f"🏆 **{local} GANARÍA el partido.**"
        color = "green"
    elif prediccion == 0:
        resultado = "🤝 **Empate previsto.**"
        color = "gray"
    else:
        resultado = f"⚠️ **{visitante} GANARÍA el partido.**"
        color = "red"

    st.markdown(f"<div style='background-color:{color};padding:1rem;border-radius:10px;color:white;font-size:18px;text-align:center'>{resultado}</div>", unsafe_allow_html=True)
