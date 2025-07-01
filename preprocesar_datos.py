import pandas as pd

# Leer el Excel original
df = pd.read_excel("BDdeLiga1.xlsx")

# Separar marcador en goles
df[['Goles Local', 'Goles Visitante']] = df['Marcador'].str.extract(r'(\d+)[–-](\d+)').astype(int)

# Crear la columna Resultado en texto (Ganó Local, Empate, Ganó Visitante)
def calcular_resultado(row):
    if row["Goles Local"] > row["Goles Visitante"]:
        return "Ganó Local"
    elif row["Goles Local"] < row["Goles Visitante"]:
        return "Ganó Visitante"
    else:
        return "Empate"

df["Resultado"] = df.apply(calcular_resultado, axis=1)

# Mapear resultado textual a número
mapa_resultado = {'Ganó Local': 0, 'Empate': 1, 'Ganó Visitante': 2}
df['Resultado Codificado'] = df['Resultado'].map(mapa_resultado)

# Eliminar filas con resultado no reconocido
df = df.dropna(subset=['Resultado Codificado'])
df['Resultado Codificado'] = df['Resultado Codificado'].astype(int)

# Guardar CSV para el modelo
df.to_csv("Liga1Proce.csv", index=False)

print("✅ Preprocesamiento completado.")
