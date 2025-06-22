# Sistema Web de Detección y Reconocimiento de Placas Vehiculares (Perú)

Este proyecto implementa un sistema web para detectar placas vehiculares peruanas usando YOLO11 y reconocer el texto con EasyOCR. Incluye preprocesamiento avanzado, corrección de perspectiva y una interfaz web moderna con Flask.

---

## 🚀 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Windows (recomendado, pero funciona en Linux/Mac con ajustes menores)

## 📦 Instalación de dependencias

1. **Clona o descarga este repositorio**
2. Abre una terminal en la carpeta del proyecto
3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

> Si tienes problemas con EasyOCR, instala también:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install easyocr
> ```

## 📁 Estructura principal

```
├── app.py                  # Backend Flask
├── requirements.txt        # Dependencias
├── models/
│   └── modeloPlacaCNN.pt   # Modelo YOLO11 entrenado
├── static/
│   ├── upload/             # Imágenes subidas
│   ├── predict/            # Imágenes con detección
│   └── roi/                # Recortes de placas
├── templates/
│   └── index.html          # Interfaz web
└── notebook-rody.ipynb     # Notebook de entrenamiento y pruebas
```

## 🖥️ Ejecución del sistema web

1. Abre una terminal en la carpeta del proyecto.
2. Ejecuta:

```bash
python app.py
```

3. Abre tu navegador y ve a [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Sube una imagen de un vehículo con placa visible. El sistema detectará la placa, la recortará, corregirá la perspectiva y mostrará el texto reconocido.

## 📝 Notas importantes

- El sistema elimina automáticamente las imágenes temporales al recargar la página.
- El modelo YOLO11n debe estar entrenado para placas peruanas (o el país que desees).
- El OCR está optimizado para placas peruanas (amarillas/azules, letras negras).
- Si tienes problemas con EasyOCR, revisa la instalación de `torch` y `easyocr`.
- Puedes ajustar los filtros y el pipeline en `app.py` para mejorar la robustez según tus imágenes.

## 👨‍💻 Autor
- Rody Uzuriaga
- Proyecto de Visión Artificial - USIL, 2025
