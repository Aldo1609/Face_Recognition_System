import face_recognition as fr
import cv2
import numpy
import os
from datetime import datetime

ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}/{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])
    
    
def codificacion(imagenes):
    lista_codificada = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificar = fr.face_encodings(imagen)[0]
        lista_codificada.append(codificar)
    return lista_codificada


def registro_asistencia(persona):
    f = open('registro.csv', 'r+')
    lista_datos = f.readlines()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])
    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M:%S')
        f.writelines(f'\n{persona}, {string_ahora}')


lista_empleados_codificada = codificacion(mis_imagenes)
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
exito, imagen = captura.read()


if not exito:
    print("[Error] No se ha podido tomar la captura")
else:
    cara_captura = fr.face_locations(imagen)
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)
    
    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracodif)
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)
        indice_coincidencia = numpy.argmin(distancias)
        
        if distancias[indice_coincidencia] > 0.6:
            print("[Error] No coincide con ninguno de nuestros empleados")
        else:
            nombre =  nombres_empleados[indice_coincidencia]
            y1, x2, y2, x1 = caraubic
            cv2.rectangle(imagen, (x1, y1),(x2,y2), (0, 255, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            registro_asistencia(nombre)
            cv2.imshow('Imagen', imagen)
            cv2.waitKey(0)