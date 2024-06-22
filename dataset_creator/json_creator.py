###Script para generar el json de dataset para entrenar la red neuronal. Toma input desde los ssc filtrados y con notes.
#  PIU v1.0 --22/06/24


import csv
import json
import os

# Parámetros del script
"""el input_csv debe tener los siguientes campos
path,file_name,title,artist,offset,bpms,music,time_signatures,tick_counts,notes,cant_notas
"""
input_csv = 'resultado_con_cant_notas.csv' 
""" output_steps_per_sgram: Ingresar la cantidad de bits por nota (5 x cantidad de pasos) para iniciar tomaremos 4 pasos por nota """
output_steps_per_sgram = 20  

 
""" sgram_size = [X, Y] : Ingresar la cantidad de pixeles de las imagenes a ingresar a la red en pixeles """
sgram_size = [64, 64]

# Función para generar el nombre de archivo con el número de secuencia
def generate_filename(base_name, index):
    return f"{base_name}_{index}.png"

# Función para procesar el CSV y generar el JSON
def generate_json_from_csv(input_csv, output_steps_per_sgram):
    try:
        with open(input_csv, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            data = {
                "output_steps_per_sgram": output_steps_per_sgram,
                "sgram_size": sgram_size,
                "sgrams_with_steps": {}
            }

            for row in reader:
                path = row['path']
                cant_notas = int(row['cant_notas'])
                notes = row['notes'].strip('[]').split(';')

                # Obtener el nombre base del archivo y cambiar extensión a .png
                base_name = os.path.splitext(os.path.basename(path))[0]

                # Generar nombres de archivo y asignar valores de notas
                for i in range(cant_notas):
                    filename = generate_filename(base_name, i)
                    data["sgrams_with_steps"][filename] = notes[i] if i < len(notes) else ""

            # Escribir el JSON en un archivo
            output_json = os.path.splitext(input_csv)[0] + '.json'
            with open(output_json, 'w', encoding='utf-8') as outfile:
                json.dump(data, outfile, indent=4)

            print(f"JSON guardado en: {output_json}")

    except Exception as e:
        print(f"Error al procesar el archivo CSV: {e}")

# Llamada a la función para generar el JSON
generate_json_from_csv(input_csv, output_steps_per_sgram)