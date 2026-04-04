"""
Genera las imágenes PNG de los marcadores Aruco para el mundo Webots.
Ejecutar una sola vez antes de lanzar la simulación:
    python3 webots/generate_aruco_markers.py
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_PX   = 300   # tamaño del marcador en píxeles
BORDER_PX   = 40    # borde blanco para mejor detección

for marker_id in range(4):
    marker = aruco.generateImageMarker(DICTIONARY, marker_id, MARKER_PX)

    # Añadir borde blanco
    total = MARKER_PX + 2 * BORDER_PX
    canvas = np.ones((total, total), dtype=np.uint8) * 255
    canvas[BORDER_PX:BORDER_PX + MARKER_PX,
           BORDER_PX:BORDER_PX + MARKER_PX] = marker

    # Espejo horizontal para corregir la inversión de textura en Webots
    canvas = cv2.flip(canvas, 1)

    path = os.path.join(OUTPUT_DIR, f'aruco_{marker_id}.png')
    cv2.imwrite(path, canvas)
    print(f"Generado: {path}")

print("Listo. Reinicia Webots para que cargue las texturas.")
