# 🧠 NeuroVISION: Detección de Tumores en Imágenes Médicas 🏥

![image](https://github.com/No-Country-simulation/NeuroVISION/blob/main/img/Portada.jpg)

### 📝 Índice:

- [**Introducción**](#introducción)
- [**Objetivo**](#objetivo)
- [**Metodología**](#metodología)
- [**Datos**](#datos)
- [**Visualización en Streamlit**](#visualización-en-streamlit)
- [**Herramientas utilizadas en el proyecto**](#herramientas-utilizadas-en-el-proyecto)
- [**Contacto**](#contacto)

---

## 🔥 Introducción 🔥

NeuroVISION es un sistema basado en **visión por computadora e inteligencia artificial** diseñado para la **detección y análisis de tumores cerebrales** en imágenes médicas.

Utilizando técnicas de **machine learning y deep learning**, la plataforma analiza imágenes de **resonancia magnética (MRI)** para identificar la **probabilidad de presencia de tumores**, facilitando el diagnóstico temprano y la toma de decisiones médicas.

Mediante un dashboard interactivo en **Streamlit**, NeuroVISION ofrece herramientas visuales de **segmentación de imágenes, estadísticas clave y generación automática de reportes** para médicos, radiólogos y oncólogos.

---

## 🔥 Objetivo 🔥

- **Detectar anomalías cerebrales** en imágenes médicas mediante técnicas de visión por computadora y machine learning.
- **Implementar modelos de segmentación y clasificación de tumores** para apoyar la toma de decisiones clínicas.
- **Evaluar características morfológicas del cráneo y tumores** mediante análisis de imágenes.
- **Optimizar la detección temprana y monitoreo de tumores** para mejorar la precisión y reducir tiempos de diagnóstico.
- **Fomentar la toma de decisiones basada en datos** mediante herramientas analíticas avanzadas.

Este proyecto está diseñado para:
- **Radiólogos y médicos especialistas**, para agilizar el diagnóstico de tumores.
- **Oncólogos**, para evaluar casos y decidir tratamientos basados en datos.
- **Investigadores médicos**, para analizar patrones en la detección de tumores.

---

## 🔥 Metodología 🔥

NeuroVISION sigue una metodología **Agile Scrum**, integrando el estándar **CRISP-DM (Cross-Industry Standard Process for Data Mining)** para garantizar una gestión estructurada del análisis de datos.

- **Preprocesamiento de imágenes**: Limpieza y conversión de imágenes médicas para su análisis.
- **Entrenamiento del modelo**: Uso de **Keras** y **OpenCV** para segmentación y clasificación binaria (tumor/no tumor).
- **Despliegue en Streamlit**: Implementación de un dashboard interactivo para visualizar resultados.
- **Generación de reportes**: Creación automática de documentos con hallazgos clave.

---

## 🔥 Datos 🔥

- **Imágenes médicas** de resonancia magnética (MRI) utilizadas para entrenar y evaluar el modelo de clasificación.
- **Conjunto de datos etiquetados** para determinar la probabilidad de presencia de tumor.

---

## 🔥 Visualización en Streamlit 🔥

- **Panel interactivo** con carga de imágenes médicas y análisis en tiempo real.
- **Segmentación automática** de regiones sospechosas en la imagen.
- **Estadísticas clave** sobre tamaño, ubicación y clasificación del tumor.
- **Exportación de reportes** en formatos accesibles para especialistas médicos.

---

## 🔥 Herramientas utilizadas en el proyecto 🔥

| Herramienta         | Logo                                     | Descripción                                                                                                           |
|---------------------|------------------------------------------|--------------------------------------------------------------------------------|
| **Python**         | <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="100" height="100">  | Lenguaje de programación principal para el procesamiento de datos. |
| **OpenCV (cv2)**   | <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg" width="100" height="100">  | Librería de visión por computadora utilizada para el análisis de imágenes. |
| **Keras**          | <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" width="100" height="100">  | Framework de deep learning utilizado para entrenar el modelo de detección. |
| **Streamlit**      | <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="100" height="100">  | Herramienta para crear el dashboard interactivo del sistema. |
| **Slack**         | <img src="https://upload.wikimedia.org/wikipedia/commons/7/76/Slack_Icon.png" width="100" height="100"> | Plataforma de comunicación en equipo. |
| **Trello**         | <img src="data:image/jpeg;base64,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" width="100" height="100">  | Gestión de tareas y planificación del proyecto. |
| **Google Drive**   | <img src="https://upload.wikimedia.org/wikipedia/commons/1/12/Google_Drive_icon_%282020%29.svg" width="100" height="100"> | Almacenamiento y sincronización de archivos del equipo. |

---

## 🔥 Contacto 🔥

| Integrantes         |                                     | Rol                                   | GitHub                                        | LinkedIn                                                                           |
|---------------------|-------------------------------------|---------------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------|
| Miguel Ismerio | <img src="https://github.com/No-Country-simulation/s18-18-t-data-bi/blob/main/img/Miguel.png" width="100" height="100" style="border-radius: 50%;">  | **Data Scientist / Project Manager** | [GitHub](https://github.com/mikeismerio) | [LinkedIn](https://www.linkedin.com/in/miguel-ismerio/)  |
| x2   | <img src="https://github.com/No-Country-simulation/s18-18-t-data-bi/blob/main/img/Antonia.jpg" width="100" height="100" style="border-radius: 50%;">      | Machine Learning Developer | [GitHub](https://github.com/asoler2004) | [LinkedIn](https://www.linkedin.com/in/antonia-soler-7a2811230)  |
| x2   | Data Scientist | [GitHub](https://github.com/Carrillo1992) | [LinkedIn](https://www.linkedin.com/in/daniel-carrillo-b04a862a2)  |
| x2     | Data Analytics | [GitHub](https://github.com/luceldasilva) | [LinkedIn](https://www.linkedin.com/in/luceldasilva/)  |

---

