import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import re

# Leer y combinar hojas del Excel
archivo = "BD de Liga1.xlsx"
xls = pd.ExcelFile(archivo)
df = pd.concat([xls.parse(hoja) for hoja in xls.sheet_names], ignore_index=True)

# Normalizar y limpiar columna 'Marcador'
df['Marcador'] = df['Marcador'].astype(str).str.replace('–', '-', regex=False)

# Función robusta para extraer los goles
def extraer_goles(marcador):
    numeros = re.findall(r'\d+', marcador)
    if len(numeros) == 2:
        return int(numeros[0]), int(numeros[1])
    else:
        return None, None

# Aplicar la función y eliminar filas inválidas
df[['Goles_Local', 'Goles_Visitante']] = df['Marcador'].apply(lambda x: pd.Series(extraer_goles(x)))
df = df.dropna(subset=['Goles_Local', 'Goles_Visitante'])

# Codificar resultado
le = LabelEncoder()
df['Resultado_Codificado'] = le.fit_transform(df['Resultado'])

# Guardar CSV limpio para entrega
df.to_csv("Liga1Proce.csv", index=False)

# Preparar variables de entrada y salida
X = df[['Goles_Local', 'Goles_Visitante']]
y = df['Resultado_Codificado']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo y codificador
joblib.dump(model, "modelo.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ CSV generado y modelo entrenado con éxito.")
