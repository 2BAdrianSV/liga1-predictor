import pandas as pd

# Cargar el Excel
archivo = "BD de Liga1.xlsx"
xls = pd.ExcelFile(archivo)

# Combinar todas las hojas
df_completo = pd.concat([xls.parse(hoja) for hoja in xls.sheet_names], ignore_index=True)

# Verifica y guarda
print(f"Datos combinados: {df_completo.shape}")
df_completo.to_csv("Liga1Proce.csv", index=False)
