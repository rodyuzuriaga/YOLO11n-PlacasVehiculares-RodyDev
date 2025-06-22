import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re

UPLOAD_FOLDER = 'static/upload/'
PREDICT_FOLDER = 'static/predict/'
ROI_FOLDER = 'static/roi/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER
app.config['ROI_FOLDER'] = ROI_FOLDER

# Cargar modelo y OCR
model = YOLO("./models/modeloPlacaCNN.pt")
reader = easyocr.Reader(['es', 'en'])

# --- Utilidades ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filtrar_placa_peru(texto):
    # Elimina saltos de línea, espacios y la palabra PERU
    texto = texto.replace('\n', '').replace('\r', '').replace(' ', '')
    texto = re.sub(r'PERU', '', texto, flags=re.IGNORECASE)
    # Solo placas con guion y letras/números
    patrones = [
        r'([A-Z]{3}-\d{3,4})',   # ABC-123 o ABC-1234
        r'([A-Z]{2}-\d{4})',     # AB-1234
        r'([A-Z]{3}\d{3,4})',   # ABC123 o ABC1234 (sin guion, fallback)
        r'([A-Z]{2}\d{4})',     # AB1234 (sin guion, fallback)
    ]
    for patron in patrones:
        match = re.search(patron, texto)
        if match:
            return match.group(1)
    return ''

def ocr_placa_canny_easyocr(plate_img, easyocr_reader):
    # Pipeline de filtros: negativo, binarización, adaptativo, original, y amarillo sobre fondo negro
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    _, binary_otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    negativo = cv2.bitwise_not(binary_otsu)
    adapt = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    # Filtro amarillo sobre fondo negro (simula placa peruana)
    amarillo = cv2.cvtColor(negativo, cv2.COLOR_GRAY2BGR)
    amarillo[np.where((negativo==255))] = (0,255,255)  # Letras amarillas
    amarillo[np.where((negativo==0))] = (0,0,0)        # Fondo negro
    # Lista de filtros a probar
    filtros = [
        (negativo, 'Negativo'),
        (binary_otsu, 'Binarización Otsu'),
        (adapt, 'Adaptativo'),
        (gray, 'Gris'),
        (amarillo, 'AmarilloNegro')
    ]
    mejor_texto = '(Sin placa detectada)'
    mejor_conf = 0
    mejor_roi = negativo
    for roi, nombre in filtros:
        # Si es color, pasar a gray para EasyOCR
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        ocr_result = easyocr.Reader(['es']).readtext(roi_gray, detail=1, paragraph=False)
        print(f"[DEBUG] EasyOCR {nombre}: {ocr_result}")
        candidatos = []
        for result in ocr_result:
            text = result[1]
            conf = result[2]
            if len(text) >= 6:
                candidatos.append((text, conf))
        placas_validas = []
        for t, c in candidatos:
            t_filtrado = filtrar_placa_peru(t)
            if t_filtrado and '-' in t_filtrado and len(t_filtrado) >= 6:
                placas_validas.append((t_filtrado, c))
        if placas_validas:
            placas_validas.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            if placas_validas[0][1] > mejor_conf:
                mejor_texto = placas_validas[0][0]
                mejor_conf = placas_validas[0][1]
                mejor_roi = roi
    print("[DEBUG] Texto final mostrado:", mejor_texto)
    return mejor_texto, mejor_roi

def preprocesar_placa_avanzado(plate_img):
    # Si la placa es pequeña, la agrandamos (super-resolución simple)
    h, w = plate_img.shape[:2]
    if w < 180 or h < 60:
        scale = max(2, int(240 / min(w, h)))
        plate_img = cv2.resize(plate_img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    # Denoising
    placa_denoise = cv2.fastNlMeansDenoisingColored(plate_img, None, 10, 10, 7, 21)
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    placa_sharp = cv2.filter2D(placa_denoise, -1, kernel)
    # Contraste y brillo
    placa_final = cv2.convertScaleAbs(placa_sharp, alpha=1.3, beta=10)
    return placa_final

def ocr_placa_multifiltro(plate_img, easyocr_reader):
    # Preprocesamiento avanzado antes de los filtros
    plate_img = preprocesar_placa_avanzado(plate_img)
    filtros = []
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    _, bin_otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filtros.append(("Otsu", bin_otsu))
    filtros.append(("Otsu_neg", cv2.bitwise_not(bin_otsu)))
    bin_adapt = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    filtros.append(("Adapt", bin_adapt))
    filtros.append(("Adapt_neg", cv2.bitwise_not(bin_adapt)))
    _, bin_fixed = cv2.threshold(gray_clahe, 180, 255, cv2.THRESH_BINARY)
    filtros.append(("Fixed", bin_fixed))
    filtros.append(("Fixed_neg", cv2.bitwise_not(bin_fixed)))
    filtros.append(("Gray", gray_clahe))
    filtros.append(("Gray_neg", cv2.bitwise_not(gray_clahe)))
    filtros.append(("Color", plate_img))
    filtros.append(("RGB", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)))
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_on_black = cv2.bitwise_and(plate_img, plate_img, mask=mask_yellow)
    yellow_on_black = cv2.cvtColor(yellow_on_black, cv2.COLOR_BGR2GRAY)
    filtros.append(("Yellow", yellow_on_black))

    resultados = []
    for nombre, img in filtros:
        ocr_result = easyocr_reader.readtext(img, detail=1, paragraph=False)
        for result in ocr_result:
            text = result[1]
            conf = result[2]
            if len(text) >= 6:
                t_filtrado = filtrar_placa_peru(text)
                if t_filtrado and '-' in t_filtrado and len(t_filtrado) >= 6:
                    resultados.append({
                        'texto': t_filtrado,
                        'conf': conf,
                        'img': img,
                        'filtro': nombre,
                        'ocr': 'EasyOCR'
                    })
    if resultados:
        resultados.sort(key=lambda x: (x['conf'], len(x['texto'])), reverse=True)
        mejor = resultados[0]
        print(f"[DEBUG] Mejor filtro: {mejor['filtro']} | OCR: {mejor['ocr']} | Texto: {mejor['texto']} | Conf: {mejor['conf']}")
        return mejor['texto'], mejor['img'], mejor['filtro'], mejor['ocr']
    else:
        print("[DEBUG] Sin placa detectada tras todos los filtros.")
        return '(Sin placa detectada)', filtros[0][1], '-', '-'

def corregir_perspectiva(placa_img):
    gray = cv2.cvtColor(placa_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return placa_img
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    def ordenar_puntos(pts):
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2][np.argsort(pts[:2, 0])]
        bottom = pts[2:][np.argsort(pts[2:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
    box = ordenar_puntos(box)
    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    if maxWidth < 10 or maxHeight < 10:
        return placa_img
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    aligned = cv2.warpPerspective(placa_img, M, (maxWidth, maxHeight))
    return aligned

# --- Rutas ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Borrar imágenes temporales al cargar la página (GET)
    if request.method == 'GET':
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PREDICT_FOLDER'], app.config['ROI_FOLDER']]:
            for f in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, f))
                except Exception:
                    pass
    upload = False
    upload_image = None
    text = None
    if request.method == 'POST':
        if 'image_name' not in request.files:
            return redirect(request.url)
        file = request.files['image_name']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        upload = True
        upload_image = filename
        # Leer imagen y predecir
        img = cv2.imread(filepath)
        results = model(img)
        img_pred = img.copy()
        roi_img = None
        ocr_result = ''
        for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            overlay = img_pred.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), -1)  # Relleno verde
            alpha = 0.35  # Transparencia
            img_pred = cv2.addWeighted(overlay, alpha, img_pred, 1 - alpha, 0)
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0,255,0), 2)  # Borde verde
            crop = img[y1:y2, x1:x2]
            roi_alineado = corregir_perspectiva(crop)
            ocr_result, roi_procesado, filtro_usado, ocr_usado = ocr_placa_multifiltro(roi_alineado, reader)
            cv2.imwrite(os.path.join(app.config['ROI_FOLDER'], filename), roi_procesado)
            roi_img = roi_procesado
            if roi_img is not None:
                break
        cv2.imwrite(os.path.join(app.config['PREDICT_FOLDER'], filename), img_pred)
        text = ocr_result
        filtro = filtro_usado
        ocr_engine = ocr_usado
    else:
        filtro = None
        ocr_engine = None
    return render_template('index.html', upload=upload, upload_image=upload_image, text=text, filtro=filtro, ocr_engine=ocr_engine)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PREDICT_FOLDER, exist_ok=True)
    os.makedirs(ROI_FOLDER, exist_ok=True)
    app.run(debug=True)
