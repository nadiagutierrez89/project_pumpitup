import pandas as pd

# Define la función para convertir CSV a Excel
def csv_to_excel(csv_file, excel_file):
    try:
        df = pd.read_csv(csv_file)
        df.to_excel(excel_file, index=False)
        print(f"Excel guardado en: {excel_file}")
    except Exception as e:
        print(f"Error al convertir CSV a Excel: {e}")

# Llama a la función para convertir el archivo CSV generado a Excel
csv_file = 'resultado_con_cant_notas.csv'
excel_file = 'resultado_filtrado_con_cant_notas.xlsx'
csv_to_excel(csv_file, excel_file)
