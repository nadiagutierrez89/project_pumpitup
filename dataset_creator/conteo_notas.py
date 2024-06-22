import csv

# Nombre de los archivos de entrada y salida
input_csv = 'resultado_filtrado.csv'
output_csv_with_count = 'resultado_con_cant_notas.csv'

# Función para contar las notas en el campo 'notes'
def count_notes_in_field(notes):
    # Remover los corchetes y dividir por el separador ';'
    cleaned_notes = notes.strip('[]')
    return len(cleaned_notes.split(';')) if cleaned_notes else 0

# Leer el CSV de entrada, agregar el campo 'cant_notas' y guardar en el nuevo CSV
def add_notes_count_to_csv(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + ['cant_notas']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in reader:
                row['cant_notas'] = count_notes_in_field(row['notes'])
                writer.writerow(row)

        print(f"CSV con conteo de notas guardado en: {output_path}")
    except Exception as e:
        print(f"Error al procesar el archivo CSV: {e}")

# Llamada a la función para procesar el archivo CSV
add_notes_count_to_csv(input_csv, output_csv_with_count)
