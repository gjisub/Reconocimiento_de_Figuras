import cv2
import numpy as np
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# ==============================
# SELECTOR DE IMAGEN
# ==============================
Tk().withdraw()

carpeta = os.path.join(os.getcwd(), "imagenes")

ruta = askopenfilename(
    initialdir=carpeta,
    title="Selecciona una imagen",
    filetypes=[("Imagenes", "*.png *.jpg *.jpeg")]
)

if not ruta:
    print("No seleccionaste imagen")
    exit()

imagen = cv2.imread(ruta)

if imagen is None:
    print("Error al cargar la imagen")
    exit()

imagen = cv2.resize(imagen, (1000, 500))
resultado = imagen.copy()

# ==============================
# PREPROCESAMIENTO
# ==============================
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imagenHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# ==============================
# FUNCIONES
# ==============================
def detectar_color(mask):
    # Extraer solo los píxeles de la figura
    h_values = imagenHSV[:,:,0][mask == 255]

    if len(h_values) == 0:
        return "Otro"

    # Obtener el valor más frecuente (modo)
    hist = np.bincount(h_values)
    h = np.argmax(hist)

    # Clasificación por H
    if (h < 10 or h > 170):
        return "Rojo"
    elif 10 <= h < 20:
        return "Naranja"
    elif 20 <= h < 40:   # 👈 amarillo bien amplio
        return "Amarillo"
    elif 40 <= h < 85:
        return "Verde"
    elif 85 <= h < 130:
        return "Azul"
    elif 130 <= h < 160:
        return "Violeta"
    elif 160 <= h <= 170:
        return "Rosa"
    else:
        return "Otro"

def distancia(p1, p2):
    return math.sqrt((p1[0][0]-p2[0][0])**2 + (p1[0][1]-p2[0][1])**2)

def detectar_figura(c):
    epsilon = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    lados = len(approx)

    # ===== TRIÁNGULO =====
    if lados == 3:
        return "Triangulo"

    # ===== CUADRILÁTEROS =====
    elif lados == 4:
        pts = approx.reshape(4,2)

        pendientes = []
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i+1)%4]
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                pendientes.append(999)
            else:
                pendientes.append(abs(dy/dx))

        alineados = sum(1 for p in pendientes if p < 0.3 or p > 3)

        if alineados >= 3:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)

            if 0.9 <= ratio <= 1.1:
                return "Cuadrado"
            else:
                return "Rectangulo"
        else:
            return "Rombo"

    # ===== OTRAS FIGURAS =====
    elif lados == 5:
        return "Pentagono"

    elif lados == 6:
        return "Hexagono"

    else:
        return "Circulo"

# ==============================
# PROCESAMIENTO
# ==============================
contador = {}

for c in contornos:

    area = cv2.contourArea(c)
    if area < 200:
        continue

    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    figura = detectar_figura(c)
    color = detectar_color(mask)

    M = cv2.moments(c)
    if M["m00"] == 0:
        continue

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    clave = figura + " " + color
    contador[clave] = contador.get(clave, 0) + 1

    cv2.drawContours(resultado, [c], -1, (0,255,0), 2)
    cv2.putText(resultado, clave, (cx-60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

# ==============================
# RESULTADOS
# ==============================
print("\n===== RESULTADO =====")
for k, v in contador.items():
    print(f"{v} {k}")

cv2.imshow("Resultado", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()