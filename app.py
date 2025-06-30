
import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("modelo.pkl")

# Título
st.title("Predicción de Resultados - Liga 1 Peruana 2025")
st.subheader("Sistema de predicción usando IA y Aprendizaje Estadístico")

# Entradas del usuario
st.markdown("### Ingrese los datos del partido")

goles_local = st.number_input("Goles esperados del equipo Local", min_value=0.0, max_value=10.0, step=0.5)
goles_visitante = st.number_input("Goles esperados del equipo Visitante", min_value=0.0, max_value=10.0, step=0.5)

# Botón para predecir
if st.button("Predecir Resultado"):
    entrada = np.array([[goles_local, goles_visitante]])
    prediccion = model.predict(entrada)[0]

    resultado = ""
    if prediccion == 1:
        resultado = "Gana el Local"
    elif prediccion == 0:
        resultado = "Empate"
    else:
        resultado = "Gana el Visitante"

    st.success(f"✅ Resultado Predicho: **{resultado}**")
