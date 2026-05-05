import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Predicción Continental", layout="wide")
st.title("🎓 Sistema de Alerta: Deserción Estudiantil")

@st.cache_resource
def cargar_todo():
    modelo = tf.keras.models.load_model('modelo_mlp.h5')
    escalador = joblib.load('escalador.pkl')
    # Cargar los nombres de las características
    feature_names = joblib.load('features_names.pkl')
    return modelo, escalador, feature_names

try:
    modelo, escalador, FEATURE_NAMES = cargar_todo()
    st.sidebar.success("✅ Modelo, escalador y features cargados correctamente")
    st.sidebar.write(f"📊 Features esperadas: {len(FEATURE_NAMES)}")
    st.sidebar.write(f"📐 Tipo de escalador: {type(escalador).__name__}")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar archivos: {e}")
    st.stop()

# --- INTERFAZ DE USUARIO ---
with st.form("mi_formulario"):
    st.subheader("📋 Ingrese los datos del alumno")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 DATOS ACADÉMICOS**")
        nota1 = st.slider("Nota Promedio - 1er Semestre", 0.0, 20.0, 10.0, step=0.5)
        nota2 = st.slider("Nota Promedio - 2do Semestre", 0.0, 20.0, 10.0, step=0.5)
        materias_aprobadas_1er = st.slider("Materias aprobadas - 1er Semestre", 0, 10, 5)
        materias_matriculadas_1er = st.slider("Materias matriculadas - 1er Semestre", 1, 10, 6)
        materias_aprobadas_2do = st.slider("Materias aprobadas - 2do Semestre", 0, 10, 5)
        materias_matriculadas_2do = st.slider("Materias matriculadas - 2do Semestre", 1, 10, 6)
        
    with col2:
        st.markdown("**👤 DATOS PERSONALES**")
        edad = st.number_input("Edad al matricularse", min_value=16, max_value=100, value=20)
        genero = st.selectbox("Sexo", options=["Mujer (0)", "Varón (1)"])
        genero_val = 1 if "Varón" in genero else 0
        estado_civil = st.selectbox("Estado Civil", [1, 2, 3, 4, 5, 6], format_func=lambda x: {1:"Soltero", 2:"Casado", 3:"Viudo", 4:"Divorciado", 5:"Unión Libre", 6:"Otro"}.get(x, "Otro"))
        
    with col3:
        st.markdown("**💰 DATOS ECONÓMICOS**")
        beca = st.selectbox("¿Tiene Beca?", options=["No (0)", "Sí (1)"])
        beca_val = 1 if "Sí" in beca else 0
        deudor = st.selectbox("¿Es deudor?", options=["No (0)", "Sí (1)"])
        deudor_val = 1 if "Sí" in deudor else 0
        matricula_al_dia = st.selectbox("¿Matrícula al día?", options=["No (0)", "Sí (1)"])
        matricula_al_dia_val = 1 if "Sí" in matricula_al_dia else 0
        pib = st.number_input("PIB (Producto Interno Bruto)", value=0.32, step=0.01, format="%.2f")

    submitted = st.form_submit_button("🔍 ANALIZAR", use_container_width=True)

if submitted:
    try:
        # Calcular variables derivadas
        tasa_variacion = nota2 - nota1
        porcentaje_aprobados_1er = materias_aprobadas_1er / (materias_matriculadas_1er + 0.01)
        porcentaje_aprobados_2do = materias_aprobadas_2do / (materias_matriculadas_2do + 0.01)
        
        # Crear diccionario con todas las características en el ORDEN CORRECTO
        # Usando los nombres estándar del dataset UCI #697
        datos_dict = {}
        
        for name in FEATURE_NAMES:
            # Notas
            if '1st sem (grade)' in name:
                datos_dict[name] = nota1
            elif '2nd sem (grade)' in name:
                datos_dict[name] = nota2
            # Materias aprobadas
            elif '1st sem (approved)' in name:
                datos_dict[name] = materias_aprobadas_1er
            elif '2nd sem (approved)' in name:
                datos_dict[name] = materias_aprobadas_2do
            # Materias matriculadas
            elif '1st sem (enrolled)' in name:
                datos_dict[name] = materias_matriculadas_1er
            elif '2nd sem (enrolled)' in name:
                datos_dict[name] = materias_matriculadas_2do
            # Feature engineering creadas en Colab
            elif 'tasa_variacion_promedio' in name:
                datos_dict[name] = tasa_variacion
            elif 'porcentaje_aprobados_1er' in name:
                datos_dict[name] = porcentaje_aprobados_1er
            elif 'porcentaje_aprobados_2do' in name:
                datos_dict[name] = porcentaje_aprobados_2do
            # Datos personales
            elif 'Age at enrollment' in name:
                datos_dict[name] = edad
            elif 'Gender' in name:
                datos_dict[name] = genero_val
            elif 'Marital Status' in name:
                datos_dict[name] = estado_civil
            # Datos económicos
            elif 'Scholarship holder' in name:
                datos_dict[name] = beca_val
            elif 'Debtor' in name:
                datos_dict[name] = deudor_val
            elif 'Tuition fees up to date' in name:
                datos_dict[name] = matricula_al_dia_val
            elif 'GDP' in name:
                datos_dict[name] = pib
            # Variables socioeconómicas (valores por defecto)
            elif "Mother's qualification" in name:
                datos_dict[name] = 3.0
            elif "Father's qualification" in name:
                datos_dict[name] = 3.0
            elif "Mother's occupation" in name:
                datos_dict[name] = 5.0
            elif "Father's occupation" in name:
                datos_dict[name] = 5.0
            elif 'Unemployment rate' in name:
                datos_dict[name] = 6.5
            elif 'Inflation rate' in name:
                datos_dict[name] = 3.7
            elif 'Admission grade' in name:
                datos_dict[name] = 120.0
            elif 'Previous qualification (grade)' in name:
                datos_dict[name] = 130.0
            elif 'Course' in name:
                datos_dict[name] = 9238.0
            elif 'Daytime/evening attendance' in name:
                datos_dict[name] = 1.0
            elif 'Nacionality' in name:
                datos_dict[name] = 1.0
            elif 'Displaced' in name:
                datos_dict[name] = 0.0
            elif 'Educational special needs' in name:
                datos_dict[name] = 0.0
            elif 'International' in name:
                datos_dict[name] = 0.0
            # Cualquier otra variable (por defecto 0)
            else:
                datos_dict[name] = 0.0
        
        # Crear DataFrame en el orden correcto
        df_usuario = pd.DataFrame([datos_dict])
        df_usuario = df_usuario[FEATURE_NAMES]
        
        with st.expander("🔍 Ver datos enviados al modelo"):
            st.dataframe(df_usuario.iloc[:, :8])
        
        # Escalar y predecir
        X_usuario_scaled = escalador.transform(df_usuario.values)
        resultado = modelo.predict(X_usuario_scaled, verbose=0)
        probabilidad = float(resultado[0][0])
        
        # Mostrar resultados
        st.divider()
        st.subheader("📊 RESULTADO DEL ANÁLISIS")
        
        col1, col2 = st.columns(2)
        with col1:
            if probabilidad > 0.6:
                st.error(f"### ❌ ALERTA: ALTO RIESGO DE DESERCIÓN")
            elif probabilidad > 0.4:
                st.warning(f"### ⚠️ RIESGO MODERADO DE DESERCIÓN")
            else:
                st.success(f"### ✅ BAJO RIESGO DE DESERCIÓN")
        with col2:
            st.metric("Probabilidad de Deserción", f"{probabilidad:.1%}")
        
        # Barra de probabilidad
        st.progress(probabilidad, text=f"Probabilidad de deserción: {probabilidad:.1%}")
        
        # Detalles para depuración
        with st.expander("📊 Ver detalles del cálculo"):
            st.write(f"Nota 1er Semestre: {nota1}")
            st.write(f"Nota 2do Semestre: {nota2}")
            st.write(f"Tasa de variación: {tasa_variacion:.2f}")
            st.write(f"Porcentaje aprobados 1er: {porcentaje_aprobados_1er:.2f}")
            st.write(f"Porcentaje aprobados 2do: {porcentaje_aprobados_2do:.2f}")
            st.write(f"Edad: {edad}")
            st.write(f"Género: {genero_val}")
            st.write(f"Beca: {beca_val}")
            st.write(f"Deudor: {deudor_val}")
            st.write(f"PIB: {pib}")
        
        # Recomendaciones
        if probabilidad > 0.6:
            st.error("""
            ### 🚨 PROTOCOLO DE INTERVENCIÓN - ALTO RIESGO
            
            **Acciones inmediatas:**
            1. 📞 Contactar al estudiante en las próximas 48 horas
            2. 📚 Programar tutorías académicas intensivas
            3. 💰 Evaluar opciones de apoyo financiero
            4. 🎯 Reunión con el departamento de consejería estudiantil
            """)
        elif probabilidad > 0.4:
            st.warning("""
            ### 📋 PROTOCOLO DE PREVENCIÓN - RIESGO MODERADO
            
            **Acciones recomendadas:**
            1. 📧 Enviar alerta preventiva al tutor asignado
            2. 📝 Invitar al estudiante a talleres de técnicas de estudio
            3. 📊 Monitoreo mensual de calificaciones
            4. 🗓️ Seguimiento cada 30 días
            """)
        else:
            st.success("""
            ### ✅ SEGUIMIENTO ESTÁNDAR - BAJO RIESGO
            
            **Acciones a realizar:**
            1. 📋 Monitoreo pasivo del rendimiento académico
            2. 🔍 Revisión periódica de calificaciones cada 60 días
            3. 📊 Reporte semestral al director de carrera
            """)
        
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")
        import traceback
        with st.expander("Ver error detallado"):
            st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.info("""
**📁 Archivos necesarios en la misma carpeta:**
- `modelo_mlp.h5` (modelo de red neuronal)
- `escalador.pkl` (escalador StandardScaler)
- `features_names.pkl` (nombres de las características)

**📌 Nota:** Si aún no tienes `features_names.pkl`, ejecuta el código de diagnóstico en Colab para generarlo.
""")