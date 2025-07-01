import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import base64

# Cambiar fondo a blanco y letras a negro, título azul
st.markdown('''
    <style>
    body, .stApp { background-color: #fff !important; color: #111 !important; }
    .main-title {font-size: 2.5rem; font-weight: bold; color: #1a237e !important; text-align: center; margin-bottom: 0.5rem;}
    .subtitle {font-size: 1.3rem; color: #3949ab; text-align: center; margin-bottom: 1rem;}
    .divider {border-top: 2px solid #3949ab; margin: 1rem 0;}
    .block-container { max-width: 1200px !important; padding-left: 3rem !important; padding-right: 3rem !important; }
    label, .stSelectbox label, .stNumberInput label, .stMetric, .stButton, .stMarkdown, .stText, .stDataFrame, .stTable, .stAlert, .stColumn {
        color: #111 !important;
    }
    .desglose-azul { color: #1a237e !important; font-weight: bold; }
    label, .stNumberInput label { color: #1a237e !important; font-weight: bold; }
    /* Personalización avanzada del selectbox y menú */
    .stSelectbox > div[data-baseweb="select"] > div {
        background-color: #fff !important;
        color: #111 !important;
        border: 2px solid #1a237e !important;
        border-radius: 8px !important;
    }
    /* Forzar el popover del menú desplegable a blanco y texto negro */
    div[data-baseweb="popover"] {
        background-color: #fff !important;
        color: #111 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px #0002 !important;
    }
    div[data-baseweb="popover"] * {
        background-color: #fff !important;
        color: #111 !important;
    }
    .stSelectbox [data-baseweb="option"] {
        background-color: #fff !important;
        color: #111 !important;
    }
    .stSelectbox [aria-selected="true"] {
        background-color: #e3eafc !important;
        color: #1a237e !important;
    }
    .stSelectbox [role="listbox"] {
        background-color: #fff !important;
        color: #111 !important;
    }
    /* Personalización de los number_input */
    .stNumberInput input {
        background-color: #fff !important;
        color: #111 !important;
        border: 2px solid #1a237e !important;
        border-radius: 8px !important;
        font-weight: bold;
    }
    .stNumberInput button {
        background-color: #e3eafc !important;
        color: #1a237e !important;
        border-radius: 8px !important;
    }
    .stNumberInput button:hover {
        background-color: #1a237e !important;
        color: #fff !important;
    }
    .stButton>button {background-color: #1a237e !important; color: #fff !important; font-weight: bold; border-radius: 8px;}
    .stButton>button:hover {background-color: #3949ab !important; color: #fff !important;}
    </style>
''', unsafe_allow_html=True)

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

# --- INTERFAZ MEJORADA CON LOGOS Y SELECCIÓN EN FILA ---
st.markdown("""
    <div class='main-title'>📊 Predicción de Resultados - Liga 1 Peruana 2025</div>
    <div class='subtitle'>Sistema de predicción usando IA y Aprendizaje Estadístico</div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)

# Selección de equipos en una sola fila con logos y VS
st.markdown("### 1️⃣ Seleccione los equipos del partido")
equipos = sorted(df["Local"].unique())
col_local, col_vs, col_visit = st.columns([4,1,4], gap="large")

with col_local:
    col_logo, col_select = st.columns([1,5], gap="small")
    with col_select:
        local = st.selectbox("Equipo Local", equipos, key="local")
    with col_logo:
        logo_local_path = f"logos/{local}.png"
        if os.path.exists(logo_local_path):
            st.image(logo_local_path, width=64)

with col_vs:
    st.markdown("<div style='text-align:center;font-size:2rem;margin-top:2.5rem;color:#111'>VS</div>", unsafe_allow_html=True)

with col_visit:
    col_select, col_logo = st.columns([5,1], gap="small")
    with col_select:
        equipos_visitante = [e for e in equipos if e != local]
        visitante = st.selectbox("Equipo Visitante", equipos_visitante, key="visitante")
    with col_logo:
        logo_visit_path = f"logos/{visitante}.png"
        if os.path.exists(logo_visit_path):
            st.image(logo_visit_path, width=64)

if local == visitante:
    st.warning("El equipo local y visitante no pueden ser el mismo. Por favor, seleccione equipos diferentes.")
    st.stop()

# Cálculo de promedios
avg_local = df[df["Local"] == local]["Goles Local"].mean()
avg_visit = df[df["Visitante"] == visitante]["Goles Visitante"].mean()

st.markdown("### 2️⃣ Estadísticas Promedio")
col1, col2 = st.columns(2)
col1.markdown(f"<span style='color:#1a237e;font-weight:bold'>Goles promedio de {local} (local)</span>", unsafe_allow_html=True)
col1.markdown(f"<span style='font-size:2.2rem;color:#1a237e;font-weight:bold'>{avg_local:.2f}</span>", unsafe_allow_html=True)
col2.markdown(f"<span style='color:#1a237e;font-weight:bold'>Goles promedio de {visitante} (visitante)</span>", unsafe_allow_html=True)
col2.markdown(f"<span style='font-size:2.2rem;color:#1a237e;font-weight:bold'>{avg_visit:.2f}</span>", unsafe_allow_html=True)

st.markdown("### <span class='desglose-azul'>3️⃣ Ajuste los goles esperados</span>", unsafe_allow_html=True)
col_gol1, col_gol2 = st.columns(2)
with col_gol1:
    goles_local = st.number_input(
        "⚽ Goles esperados del equipo Local",
        min_value=0, max_value=10, step=1, value=int(round(avg_local)), key="goles_local"
    )
with col_gol2:
    goles_visitante = st.number_input(
        "⚽ Goles esperados del equipo Visitante",
        min_value=0, max_value=10, step=1, value=int(round(avg_visit)), key="goles_visitante"
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Botón para predecir y mostrar estadística circular
st.markdown("### 4️⃣ Resultado de la predicción")
if st.button("🔮 Predecir Resultado", use_container_width=True):
    entrada = np.array([[goles_local, goles_visitante]])
    prediccion = model.predict(entrada)[0]
    proba = model.predict_proba(entrada)[0] if hasattr(model, 'predict_proba') else [0,0,0]

    # Asumimos: 0=Empate, 1=Gana Local, 2=Gana Visita
    etiquetas = [f"Empate", f"{local}", f"{visitante}"]
    colores = ["#616161", "#388e3c", "#d32f2f"]
    if len(proba) == 3:
        proba_dict = dict(zip(model.classes_, proba))
        proba = [proba_dict.get(0,0), proba_dict.get(1,0), proba_dict.get(2,0)]
    else:
        proba = [0,0,0]
        proba[prediccion] = 1

    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64

    def donut(prob, color, label):
        fig, ax = plt.subplots(figsize=(2,2))
        wedges, texts, autotexts = ax.pie(
            [prob, 1-prob], labels=["", ""], colors=[color, "#eee"], startangle=90,
            autopct=lambda p: f'{p:.1f}%' if p > 0 and p <= 100 else '',
            wedgeprops=dict(width=0.4)
        )
        ax.set(aspect="equal")
        plt.setp(autotexts, size=14, weight="bold", color="white")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        return f"<div style='text-align:center'><img src='data:image/png;base64,{img_b64}' width='80'/><br><span style='color:{color};font-weight:bold'>{label}</span></div>"

    col_local, col_empate, col_visit = st.columns([4,2,4], gap="large")
    with col_local:
        st.markdown(donut(proba[1], "#388e3c", f"{proba[1]*100:.1f}% {local}"), unsafe_allow_html=True)
    with col_empate:
        st.markdown(donut(proba[0], "#616161", f"{proba[0]*100:.1f}% Empate"), unsafe_allow_html=True)
    with col_visit:
        st.markdown(donut(proba[2], "#d32f2f", f"{proba[2]*100:.1f}% {visitante}"), unsafe_allow_html=True)

    # Mensaje destacado
    if prediccion == 1:
        resultado = f"🏆 <b>{local}</b> GANARÍA el partido."
        color = "#388e3c"
    elif prediccion == 0:
        resultado = "🤝 <b>Empate previsto.</b>"
        color = "#616161"
    else:
        resultado = f"⚠️ <b>{visitante}</b> GANARÍA el partido."
        color = "#d32f2f"

    st.markdown(
        f"""
        <div style='background-color:{color};padding:1.5rem 1rem;border-radius:12px;color:white;font-size:1.3rem;text-align:center;font-weight:bold;box-shadow:0 2px 8px #0002;margin-bottom:1rem;'>
            {resultado}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.balloons()

    # Mostrar precisión del modelo
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Evaluar en el conjunto de prueba
X_eval = df[["Goles Local", "Goles Visitante"]]
y_eval = df["Resultado Codificado"]
_, X_test, _, y_test = train_test_split(X_eval, y_eval, test_size=0.3, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown("### 📈 Evaluación del Modelo")
st.markdown(f"<span style='color:#1a237e;font-weight:bold'>Precisión del modelo:</span> <span style='font-size:1.5rem;color:#1a237e;font-weight:bold'>{accuracy*100:.2f}%</span>", unsafe_allow_html=True)

# Mostrar matriz de confusión con botón opcional
if st.checkbox("🔍 Mostrar matriz de confusión"):
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.markdown("#### Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ganó Local", "Empate", "Ganó Visitante"],
                yticklabels=["Ganó Local", "Empate", "Ganó Visitante"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

# Verificar balance de clases
conteo_clases = np.bincount(df["Resultado Codificado"])
etiquetas = ["Ganó Local", "Empate", "Ganó Visitante"]

st.markdown("### ⚖️ Distribución de Clases")
for i, count in enumerate(conteo_clases):
    st.markdown(f"🔹 <b>{etiquetas[i]}</b>: {count} partidos", unsafe_allow_html=True)

# Mensaje si hay desbalance notorio
total = sum(conteo_clases)
porcentajes = [count / total for count in conteo_clases]
max_porcentaje = max(porcentajes)
if max_porcentaje > 0.6:
    st.warning("⚠️ Atención: El dataset está desbalanceado. Una clase representa más del 60% de los datos.")

    
    st.markdown("### 📌 Valores únicos en la columna Resultado")
st.write(df["Resultado"].value_counts())

# Pie de página
st.markdown("""
    <hr style='border-top:1px solid #bbb;margin-top:2rem;margin-bottom:0.5rem;'>
    <div style='text-align:center;color:#888;font-size:0.95rem;'>
        Desarrollado por <b>Salirrosas Jhordy</b> | Proyecto IA - Liga 1 Peruana 2025
    </div>
    """, unsafe_allow_html=True)
