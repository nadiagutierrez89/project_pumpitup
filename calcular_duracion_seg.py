import csv
import os
from mutagen import File

# Función para calcular la duración de un archivo de audio en segundos usando la metadata
def get_audio_duration(audio_path):
    try:
        # Normalizar la ruta del archivo para evitar problemas con las barras invertidas
        audio_path = os.path.normpath(audio_path)
        if os.path.exists(audio_path):
            audio = File(audio_path)
            if audio is not None and audio.info is not None:
                return audio.info.length  # Duración en segundos
            else:
                print(f"Error: No se pudo leer la metadata del archivo {audio_path}")
                return "Error"
        else:
            print(f"Error: El archivo {audio_path} no existe.")
            return "Error"
    except Exception as e:
        print(f"Error al procesar el archivo de audio {audio_path}: {e}")
        return "Error"

# Función para procesar el CSV y agregar el campo de duración
def add_duration_to_csv(input_csv, output_csv):
    try:
        with open(input_csv, 'r', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + ['duracion_sec']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in reader:
                audio_path = row['path']
                row['duracion_sec'] = get_audio_duration(audio_path)
                writer.writerow(row)
        
        print(f"CSV con duración guardado en: {output_csv}")

    except Exception as e:
        print(f"Error al procesar el archivo CSV: {e}")

# Parámetros del script
input_csv = 'resultado_con_cant_notas.csv'
output_csv = 'resultado_con_cant_notas_con_duracion.csv'

# Llamada a la función para agregar la duración al CSV
add_duration_to_csv(input_csv, output_csv)
