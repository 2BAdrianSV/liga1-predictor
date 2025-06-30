# ⚽ Predicción de Resultados - Liga 1 Peruana 2025

Este proyecto implementa un sistema de predicción de resultados de fútbol utilizando técnicas de Aprendizaje Estadístico e Inteligencia Artificial. Fue desarrollado como parte del curso de **Aprendizaje Estadístico** en la UPAO.

## 📌 ¿Qué hace?

- Toma como entrada los goles esperados del equipo local y visitante
- Usa un modelo de clasificación (Random Forest)
- Predice si gana el local, empatan o gana el visitante
- La interfaz fue desarrollada con Streamlit

## 🛠️ Requisitos

Instala los paquetes necesarios con:

```bash
pip install -r requirements.txt
```

## 🚀 Cómo ejecutar la app

1. Asegúrate de tener Python 3.10 o superior instalado
2. Activa tu entorno virtual:

```bash
.\.venv\Scriptsctivate
```

3. Ejecuta la app:

```bash
streamlit run app.py
```

## 📁 Archivos principales

- `app.py`: aplicación web en Streamlit
- `train_model.py`: script para entrenar el modelo
- `modelo.pkl`: modelo entrenado
- `Liga1Proce.csv`: dataset de entrenamiento

## 👥 Autor

Jhordy y equipo – Universidad Privada Antenor Orrego – 2025
