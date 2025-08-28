# 🫀 Heart Disease Predictor

Aplicación desarrollada en **Python** y **Streamlit** que predice la probabilidad de padecer **enfermedad cardíaca** a partir de datos clínicos.  
El modelo fue entrenado con **Machine Learning** usando un conjunto de datos real y técnicas de preprocesamiento, evaluación y optimización.

> ⚠️ **Nota importante:** Esta herramienta es de **apoyo** y **no sustituye** el criterio médico profesional.

---

## 🖼️ Vista previa

### 🏠 Formulario de entrada
![Formulario de ingreso de datos](images/formulario.png)

### 📊 Ejemplo de predicción
![Resultado de predicción](images/resultado.png)

---

## 📓 Notebook del modelo

Puedes revisar todo el flujo de análisis y entrenamiento en Google Colab:  

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uNkNtCk_kXt6TkLvwRP6hxHtHlPyqzXx#scrollTo=1lxepudAcJ1t)

---

## ⚙️ Instalación local

Clona el repositorio y ejecuta la app en tu máquina:

```bash
# Clonar el proyecto
git clone https://github.com/felipegarcia123/heart-disease-predictor.git

cd heart-disease-predictor

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run app.py
