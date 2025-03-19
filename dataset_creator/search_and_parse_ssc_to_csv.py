###Script para buscar Name, BPM, Offset y Pasos (measures) en Songs SSC de PIU v1.1 --21/06/24
import os
import re
import csv

###Parámetros
# Define la ruta de la carpeta "songs" y el archivo CSV de salida
songs_directory = 'D:\Songs'
output_csv = 'resultados.csv'
filtered_csv = 'resultado_filtrado.csv'

# Define las expresiones regulares para cada campo
regex_patterns = {
    'title': re.compile(r'#TITLE:\s*([^;]+)\s*;'),
    'artist': re.compile(r'#ARTIST:\s*([^;]+)\s*;'),
    'offset': re.compile(r'#OFFSET:\s*([-+]?\d*\.\d+|\d+)\s*;'),
    'bpms': re.compile(r'#BPMS:\s*0\.000=([-+]?\d*\.\d+|\d+)\s*;'),
    'music': re.compile(r'#MUSIC:\s*([^;]+)\s*;'),
    'time_signatures': re.compile(r'#TIMESIGNATURES:\s*0\.000=([^\n;]+)\s*;'),
    'tick_counts': re.compile(r'#TICKCOUNTS:\s*0\.000=([^\n;]+)\s*;')
}

# Expresión regular para capturar el campo #STEPSTYPE:pump-single;
regex_steptype = re.compile(r'#STEPSTYPE:pump-single;')

# Expresión regular para capturar el campo #NOTES:
regex_notes = re.compile(r'#NOTES:\s*([^;]+);')

# Define la función para limpiar valores eliminando "," y ";"
def clean_value(value):
    return value.replace(',', '').replace(';', '').strip()

# Define la función para procesar cada archivo .ssc
def process_ssc_file(file_path):
    fields = {'file_name': clean_value(os.path.splitext(os.path.basename(file_path))[0])}
    try:
        # Obtener el directorio padre del archivo .ssc
        directory_path = os.path.dirname(file_path)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Buscar #STEPSTYPE:pump-single;
            match_steptype = regex_steptype.search(content)
            if match_steptype:
                # Buscar #NOTES: después de #STEPSTYPE:pump-single;
                start_index = match_steptype.end()
                content_after_steptype = content[start_index:]
                match_notes = regex_notes.search(content_after_steptype)
                if match_notes:
                    notes_block = match_notes.group(1)
                    # Verificar si el bloque de notas contiene caracteres no deseados
                    if '|' in notes_block or '}' in notes_block or 'M' in notes_block:
                        fields['notes'] = 'Notas con simbolos'
                    else:
                        # Eliminar líneas que contienen la palabra "measure"
                        notes_lines = notes_block.split('\n')
                        filtered_lines = [line for line in notes_lines if 'measure' not in line]
                        cleaned_notes = ''.join(filtered_lines)
                        
                        # Eliminar saltos de línea y comas, separar cada 20 números con ;
                        cleaned_notes = cleaned_notes.replace('\n', '').replace(',', '').strip()
                        grouped_notes = [cleaned_notes[i:i+20] for i in range(0, len(cleaned_notes), 20)]
                        formatted_notes = '[' + ';'.join(grouped_notes) + ']'
                        fields['notes'] = formatted_notes
                else:
                    fields['notes'] = 'No se encontro dato'
            else:
                fields['notes'] = 'No se encontro dato'

            # Capturar otros campos regulares
            for field, pattern in regex_patterns.items():
                match = pattern.search(content)
                value = match.group(1) if match else 'No se encontro dato'
                if field in ['path','file_name', 'title', 'artist', 'offset', 'bpms', 'music', 'time_signatures', 'tick_counts', 'notes']:
                    value = clean_value(value)
                    if field == 'music' and value.startswith('../'):
                        value = value[3:]  # Eliminar '../' al inicio si existe
                    fields[field] = value

            # Construir el campo 'path' utilizando 'music'
            if 'music' in fields:
                music_value = fields['music']
                if music_value:
                    if music_value.startswith('../'):
                        music_value = music_value[3:]  # Eliminar '../' al inicio si existe
                    fields['path'] = os.path.normpath(os.path.join(directory_path, music_value))

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
            fieldnames = ['path','file_name', 'title', 'artist', 'offset', 'bpms', 'music', 'time_signatures', 'tick_counts', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
                print(row)

        print(f"CSV guardado en: {output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")


# Procesa los archivos y guarda los resultados
results = find_and_process_ssc_files(songs_directory)
save_to_csv(results, output_csv)

# Función para filtrar los resultados
def filter_results(data):
    filtered_data = []
    for row in data:
        if not any(row[field] in ["No se encontro dato", "Notas con simbolos"] for field in ['bpms', 'offset', 'path', 'notes']):
            filtered_data.append(row)
    return filtered_data

# Función para guardar los resultados filtrados en un archivo CSV
def save_filtered_to_csv(data, output_path):
    if not data:
        print("No hay datos para guardar en el CSV filtrado.")
        return

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['path','file_name', 'title', 'artist', 'offset', 'bpms', 'music', 'time_signatures', 'tick_counts', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

        print(f"CSV filtrado guardado en: {output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV filtrado: {e}")

# Filtrar los resultados y guardar en el CSV filtrado
filtered_results = filter_results(results)
save_filtered_to_csv(filtered_results, filtered_csv)
