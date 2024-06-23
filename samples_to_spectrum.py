import csv

entry_data = []

# Open the CSV file
with open('resultado_filtrado_con_cant_notas.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file, delimiter=';')
    
    # Skip the header
    next(csv_reader)
    
    # Iterate over each row in the CSV
    for row in csv_reader:
        # Append the row to the entry_data list
        entry_data.append(row)

path,file_name,title,artist,offset,bpms,music,time_signatures,tick_counts,notes,cant_notas = entry_data[0]
print(f"Título: {title}\nArtista: {artist}\nOffset: {offset}\nBPMS: {bpms}\nMúsica: {music}\nTime Signatures: {time_signatures}\nTick Counts: {tick_counts}\nNotas: {notes}\nCantidad de notas: {cant_notas}")



import librosa
import numpy as np

def calcular_cantidad_de_compases(bpm, offset, y, sr):
    duración_offset = librosa.get_duration(y=y, sr=sr) + offset
    return duración_offset * bpm / 60 / 4

def obtener_frames_por_compas(bpm, offset, y, sr):
    # Calcular la duración de un beat en segundos
    beat_duration = 60.0 / bpm
    
    # Crear una lista para almacenar los tiempos de los beats de a 4
    beat_times = []
    
    # Inicializar el tiempo del primer beat teniendo en cuenta el offset
    current_time = offset
    
    duration = librosa.get_duration(y=y, sr=sr)

    # Calcular los tiempos de los beats hasta el final del audio
    while current_time < duration:
        beat_times.append(current_time)
        current_time += beat_duration * 4
    
    # Convertir los tiempos de los beats a frames
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    
    # Agrupar los frames en compases de 4 beats cada uno
    # compases = [beat_frames[i:i + 4] for i in range(0, len(beat_frames), 4)]
    
    return beat_frames


import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft
import matplotlib.pyplot as plt
import librosa
import os

SAMPLES_LOCATION = '/home/someone/Descargas/Songs/'

cont_encontradas = 0
cont_no_encontradas = 0

que_no_cumplen = []

for path2,file_name,title,artist,offset,bpms,music,time_signatures,tick_counts,notes,cant_notas in entry_data:

    path = path2.replace('D:\\Songs\\', '').replace("\\","/")
    sound_filepath = SAMPLES_LOCATION + path
    bpm = float(bpms.replace(',', '.')) or float(bpms)
    offset = float(offset.replace(',', '.')) 

    try:

        # Cargar archivo de audio
        y, sr = librosa.load(sound_filepath, duration=None)
    
    except FileNotFoundError:
        cont_no_encontradas += 1
        continue
    
    cont_encontradas += 1

    # Detecta el ritmo y los latidos
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Convierte marcos de tiempo a segundos
    # beat_times = librosa.frames_to_time(beats, sr=sr)

    # un "compas" se define como 4 latidos consecutivos (4/4)
    # por lo tanto hay que recorrer de 4 en 4 los latidos
    # compases = [beats[i:i+4] for i in range(0, len(beats), 4)]

    # cant_compases = calcular_cantidad_de_compases(bpm, offset, y, sr, cant_notas)
    # if cant_compases < int(cant_notas):
        # que_no_cumplen.append(path2)

    # Dibujar el espectrograma para los primeros 5 compases
    i = -1
    frames_por_compas = obtener_frames_por_compas(bpm, offset, y, sr)
    for frame in frames_por_compas:
        i += 1
        
        # Convertir los frames a muestras
        start_sample = librosa.frames_to_samples(frame)
        try:
            end_sample = librosa.frames_to_samples(frames_por_compas[i+1])
        except IndexError:
            end_sample = len(y)

        # Obtener el segmento de audio correspondiente a los frames
        y_segment = y[start_sample:end_sample]

        # Calcular un tamaño de ventana adecuado
        n_fft = min(len(y_segment), 2048)

        if not n_fft:
            print('WARNING: ', title, artist)
            continue

        # Aplicar la STFT al segmento de audio con el tamaño de ventana ajustado
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment, n_fft=n_fft)), ref=np.max)

        # Dibujar el espectrograma
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis=None, y_axis=None)
        # plt.show()

        # Guardar el espectrograma en un archivo PNG
        basename = music.split('.')[0]
        filename = f'output/{basename}_{i}.png' 
        plt.savefig(filename)
        plt.close()


print(f"No encontradas: {cont_no_encontradas}\n Enconradas: {cont_encontradas}")

