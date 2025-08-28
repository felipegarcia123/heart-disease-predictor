# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

@st.cache_resource
def load_artifacts():
    pipe = joblib.load("heart_disease_pipeline.pkl")
    with open("feature_schema.json", "r") as f:
        schema = json.load(f)
    return pipe, schema

pipe, schema = load_artifacts()
feature_order = schema["feature_order"]
maps = schema["categorical_maps"]

st.title("🫀 Heart Disease Prediction")
st.markdown("Ingresa los datos del paciente y obtén la probabilidad de **enfermedad cardíaca**.")

# ===== UI (Formulario) =====
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad (years)", min_value=18, max_value=120, value=54)
        sex = st.selectbox("Sexo", options=list(maps["sex"].keys()),
                           format_func=lambda k: maps["sex"][str(k)] if isinstance(list(maps["sex"].keys())[0], str) else maps["sex"][k])
        cp = st.selectbox("Tipo de dolor en el pecho (cp)", options=list(maps["cp"].keys()),
                          format_func=lambda k: maps["cp"][str(k)] if isinstance(list(maps["cp"].keys())[0], str) else maps["cp"][k])
        trestbps = st.number_input("Presión arterial en reposo (trestbps)", min_value=60, max_value=240, value=130)
        chol = st.number_input("Colesterol sérico (chol)", min_value=80, max_value=700, value=240)
        fbs = st.selectbox("Glucosa en ayunas >120 mg/dl (fbs)", options=list(maps["fbs"].keys()),
                           format_func=lambda k: maps["fbs"][str(k)] if isinstance(list(maps["fbs"].keys())[0], str) else maps["fbs"][k])
        restecg = st.selectbox("ECG en reposo (restecg)", options=list(maps["restecg"].keys()),
                               format_func=lambda k: maps["restecg"][str(k)] if isinstance(list(maps["restecg"].keys())[0], str) else maps["restecg"][k])
        thalach = st.number_input("Frecuencia cardíaca máx (thalach)", min_value=60, max_value=250, value=150)

    with col2:
        exang = st.selectbox("Angina inducida por ejercicio (exang)", options=list(maps["exang"].keys()),
                             format_func=lambda k: maps["exang"][str(k)] if isinstance(list(maps["exang"].keys())[0], str) else maps["exang"][k])
        oldpeak = st.number_input("Oldpeak (ST depres.)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
        slope = st.selectbox("Pendiente del ST (slope)", options=list(maps["slope"].keys()),
                             format_func=lambda k: maps["slope"][str(k)] if isinstance(list(maps["slope"].keys())[0], str) else maps["slope"][k])
        ca = st.number_input("Nº vasos coloreados (ca)", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thal", options=list(maps["thal"].keys()),
                            format_func=lambda k: maps["thal"][str(k)] if isinstance(list(maps["thal"].keys())[0], str) else maps["thal"][k])
        smoking = st.selectbox("Fumador", options=list(maps["smoking"].keys()),
                               format_func=lambda k: maps["smoking"][str(k)] if isinstance(list(maps["smoking"].keys())[0], str) else maps["smoking"][k])
        diabetes = st.selectbox("Diabetes", options=list(maps["diabetes"].keys()),
                                format_func=lambda k: maps["diabetes"][str(k)] if isinstance(list(maps["diabetes"].keys())[0], str) else maps["diabetes"][k])
        bmi = st.number_input("IMC (bmi)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    submitted = st.form_submit_button("Predecir")

# ===== Predicción =====
if submitted:
    # Construye el DF en el MISMO orden de features usado en entrenamiento
    input_dict = {
        "age": age, "sex": int(sex), "cp": int(cp), "trestbps": trestbps, "chol": chol,
        "fbs": int(fbs), "restecg": int(restecg), "thalach": thalach, "exang": int(exang),
        "oldpeak": float(oldpeak), "slope": int(slope), "ca": int(ca), "thal": int(thal),
        "smoking": int(smoking), "diabetes": int(diabetes), "bmi": float(bmi)
    }

    # Alinea al orden esperado
    x_df = pd.DataFrame([[input_dict[c] for c in feature_order]], columns=feature_order)

    # Predicción
    proba = pipe.predict_proba(x_df)[0, 1]
    pred = int(pipe.predict(x_df)[0])

    st.subheader("Resultado")
    st.metric(
        label="Probabilidad de Enfermedad Cardíaca",
        value=f"{proba*100:.1f}%",
        delta=None
    )

    st.write("**Diagnóstico:**", "🛑 Enfermedad" if pred==1 else "✅ Sin Enfermedad")
    st.caption("Nota: herramienta de apoyo, no reemplaza criterio clínico.")
