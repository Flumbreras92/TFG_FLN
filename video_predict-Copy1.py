#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os
from os.path import isfile, join
import re
import h5py
import random
import numpy as np
import h5py

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, Dropout,  Activation, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D
#from keras.utils import multi_gpu_model

from keras.optimizers import SGD
from keras.layers.merge import concatenate, add

#from keras.utils.io_utils import HDF5Matrix
from keras.utils import plot_model
from keras import optimizers

import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from random import shuffle
from scipy.ndimage import gaussian_filter


# In[2]:


def video_predict(video_input, video_output, sigma, limit):
    
    # cargamos la estructura de la red y sus pesos  
    dirmodel = '/mnt/MD1200A/lconcha/videos/Modelos/Data_augmentation/'
    
    with open(dirmodel + "data_augmentation.json") as json_file:
        json_config = json_file.read()
    model = keras.models.model_from_json(json_config)

    checkpoint_path = "/mnt/MD1200A/lconcha/videos/Modelos/Data_augmentation/training2/cp-0012.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_path)
    
    
    #Cargamos el video que queremos meter como input a la red y realizamos una prediccion
    cap = cv2.VideoCapture(video_input)
    frames = [] #almacenamos los frames del video


    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            resize = cv2.resize(frame, (128,128), interpolation = cv2.INTER_AREA)
            frames.append(resize)

        if ret == False:
            break

    it = 0
    etiq = [] #almacenaremos tan

    for line in frames: 
        etiq.append(it)
        it+=1


    groups8 = []
    boolean= False
    pos = 0

    while boolean == False:

        for line in etiq:
            if line== (0 + pos) or line == (4 + pos) or line == (8 + pos) or line == (12 + pos) or line == (16 + pos) or line == (20 + pos) or line == (24 + pos) or line == (28 + pos):
                groups8.append(frames[line])


        pos += 1
        if pos == len(frames) - 32:
            boolean = True

    sets = [] #En esta lista vamos a separar en conjuntos de 8 los frames obtenidos anteriormente
    X_values = [] #En esta lista cada conjunto de 8 frames lo convertiremos en un array de shape [8 128 128]

    for i in range(0, len(groups8), 8):
        sets.append(groups8[i:i+8])

    for j in sets: #recorremos sets y concatenamos los 8 frames, para añadirlos a una nueva lista
        X = np.stack(j)
        X_values.append(X)


    x_array = np.array(X_values)

    prediction = model.predict(x_array)

    array_prediction = np.argmax(prediction, axis=1)
    
    
    # Una vez hecha la prediccion, vemos las posiciones en las que la red ha determinado que la rata estaba en estado ataque
    pos_at = []

    posicion = 0
    for i in array_prediction:
        if i == 1:
            pos_at.append(posicion)
        posicion +=1

    b = []
    for i in range(32):
        b.append((pos_at[-1])+(i+1))

    c = pos_at + b
    c = np.array(c)
    
    #Creamos un nuevo video en el que se ven los frames etiquetados con el estado correspondiente de la rata
    
    fps = 10.0 #Frames por segundo en la reproduccion
    
    video_label = cv2.VideoWriter(video_output + '.mp4' , cv2.VideoWriter_fourcc(*"XVID"),fps,(720,720))

    cap = cv2.VideoCapture(video_input)

    it = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            if it == len(prediction):
                it -=1

            if it in c:
                resize = cv2.resize(frame, (720,720), interpolation = cv2.INTER_AREA)
                text = cv2.putText(resize, "Estado: Ataque" + ' ' + str(round((prediction[it])[1]*100,2)) +'%', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                video_label.write(text)
            else:
                resize = cv2.resize(frame, (720,720), interpolation = cv2.INTER_AREA)
                text = cv2.putText(resize, "Estado: Reposo" + ' ' + str(round((prediction[it])[0]*100,2))+'%', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                video_label.write(text)


            it +=1
        if ret == False:
            break

    video_label.release()
    
    
    
    ################################################# GAUSSIAN FILTER   ##############################################
    
    array_prediction = array_prediction.astype(np.float)
    
    filter_gaus = gaussian_filter(array_prediction, sigma)
    
    pos_at_gaus = []

    posicion = 0
    for i in filter_gaus:
        if i > limit:
            pos_at_gaus.append(posicion)
        posicion +=1

    b_gaus = []
    for i in range(32):
        b_gaus.append((pos_at_gaus[-1])+(i+1))

    c_gaus = pos_at_gaus + b_gaus
    c_gaus = np.array(c_gaus)

      
    video_label = cv2.VideoWriter(video_output + 'gaussianFilter.mp4', cv2.VideoWriter_fourcc(*"XVID"),fps,(720,720))

    cap = cv2.VideoCapture(video_input)
    
    frame_atack = []
    it = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if it in c_gaus:
                resize = cv2.resize(frame, (720,720), interpolation = cv2.INTER_AREA)
                text = cv2.putText(resize, "Estado: Ataque", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                video_label.write(text)
                frame_atack.append(it)
                
            else:
                resize = cv2.resize(frame, (720,720), interpolation = cv2.INTER_AREA)
                text = cv2.putText(resize, "Estado: Reposo", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                video_label.write(text)

            it +=1
        if ret == False:
            break

    video_label.release()
    
##################################### archivo csv ######################


#COMPROBAMOS NUMERO DE ATAQUES

    consec = np.diff(frame_atack) #comprueba si son correlativos los numeros
    
    posicion = []
    pos = 0 
    for i in consec:
        if i !=1: #si algun numero es distinto a 1 quiere decir que se pierde la correlacion con el siguiente
            posicion.append(pos)
        pos +=1
        
    #en el caso de que el video solo disponga de un ataque:
    if len(posicion) == 0:
        print('Detectado un ataque')
        
        start = frame_atack[0]/fps
        end = frame_atack[-1]/fps

        minutes_start = int(start/60)
        seconds_start = start%60

        minutes_end = int(end/60)
        seconds_end = end%60

        it = 0
        pos = []
        for i in video_input:
            if i == '/':
                pos.append(it)
            it += 1

        name = video_input[(pos[-1])+1:]

        text = [name, str(minutes_start) + ':' + str(round(seconds_start)), str(minutes_end) + ':' + str(round(seconds_end))]

        header = ['Video Name','START', 'END']

        with open(video_output + '.csv', 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(text)
            
        print('Documento creado')
    
    
    
    #En el caso de que tenga mas de un ataque:
    nuevo_array = []
    if len(posicion) >= 1:
        print('Detectados varios ataques')
        for i in posicion:
            nuevo_array.append(frame_atack[0])
            nuevo_array.append(frame_atack[i])
            nuevo_array.append(frame_atack[i+1])
            nuevo_array.append(frame_atack[-1])
            
        nuevo_array = set(nuevo_array) #quitamos duplicados
        nuevo_array=sorted(list(nuevo_array)) #ordena y lo transforma en lista
        
        it = 0
        pos = []
        for i in video_input:
            if i == '/':
                pos.append(it)
            it += 1

        name = video_input[(pos[-1])+1:]
        
        header = ['Video Name','START', 'END']

        with open(video_output + '.csv', 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range (0, len(nuevo_array), 2):
                start = nuevo_array[i]/fps
                end = nuevo_array[i+1]/fps

                minutes_start = int(start/60)
                seconds_start = start%60

                minutes_end = int(end/60)
                seconds_end = end%60

                text = [name, str(minutes_start) + ':' + str(round(seconds_start)), str(minutes_end) + ':' + str(round(seconds_end))]
                writer.writerow(text)
                
        print('Documento creado')
            
    print('Videos creados')


# In[ ]:


a = input ('Parametro 1. Inserte directorio del video de entrada: ')
b = input ('Parametro 2. Inserte directorio de los videos de salida: ')
c = input ('Parametro 3. Inserte valor sigma: ')
d = input ('Parametro 4. Inserte limite: ')


# In[6]:


video_predict(a, b, c, d)

