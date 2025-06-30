import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Cargar el dataset
df = pd.read_csv("Liga1Proce.csv")

# Selección de variables
X = df[['Goles Local', 'Goles Visitante']]
y = df['Resultado Codificado']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, "modelo.pkl")

print("✅ Modelo entrenado y guardado como modelo.pkl")
