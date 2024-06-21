###Script para buscar Name, BPM, Offset y Pasos (measures) en Songs SSC de PIU v1.0 --21/06/24
import os
import re
import csv

# Define las expresiones regulares para cada campo
regex_patterns = {
    'title': re.compile(r'#TITLE:\s*([^;]+)\s*;'),
    'artist': re.compile(r'#ARTIST:\s*([^;]+)\s*;'),
    'offset': re.compile(r'#OFFSET:\s*([-+]?\d*\.\d+|\d+)\s*;'),
    'bpms': re.compile(r'#BPMS:\s*0\.000=([-+]?\d*\.\d+|\d+)\s*;'),
    'music': re.compile(r'#MUSIC:\s*([^;]+)\s*;')
}

# Define la función para limpiar valores eliminando "," y ";"
def clean_value(value):
    return value.replace(',', '').replace(';', '').strip()

# Define la función para procesar cada archivo .ssc
def process_ssc_file(file_path):
    fields = {'file_name': clean_value(os.path.splitext(os.path.basename(file_path))[0])}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            for field, pattern in regex_patterns.items():
                match = pattern.search(content)
                value = match.group(1) if match else 'No se encontró dato'
                if field in ['title', 'artist']:
                    value = clean_value(value)
                fields[field] = value
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        fields.update({field: 'Error al leer el archivo' for field in regex_patterns})
    return fields

# Define la función para buscar archivos .ssc y procesarlos
def find_and_process_ssc_files(directory):
    if not os.path.exists(directory):
        print(f"Error: El directorio '{directory}' no existe.")
        return []
    
    results = []
    ssc_found = False
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ssc'):
                ssc_found = True
                file_path = os.path.join(root, file)
                fields = process_ssc_file(file_path)
                results.append(fields)
                break  # Solo procesa el primer .ssc encontrado en cada subcarpeta
    
    if not ssc_found:
        print(f"Error: No se encontraron archivos .ssc en el directorio '{directory}'.")
    
    return results

# Define la función para guardar los resultados en un archivo CSV
def save_to_csv(data, output_path):
    if not data:
        print("No hay datos para guardar en el CSV.")
        return

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_name', 'title', 'artist', 'offset', 'bpms', 'music']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
                print(row)
        print(f"CSV guardado en: {output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")

# Define la ruta de la carpeta "songs" y el archivo CSV de salida
songs_directory = 'D:\Songs'
output_csv = 'resultados.csv'

# Procesa los archivos y guarda los resultados
results = find_and_process_ssc_files(songs_directory)
save_to_csv(results, output_csv)
