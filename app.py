from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@600;800&display=swap');

.stApp {
    background-color: #060B10;
    background-image:
        linear-gradient(rgba(0,255,209,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,209,0.06) 1px, transparent 1px),
        radial-gradient(ellipse at center, rgba(0,255,209,0.05) 0%, transparent 70%);
    background-size: 28px 28px, 28px 28px, 100% 100%;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #00FFD1 !important;
    text-shadow: 0 0 8px rgba(0,255,209,0.5);
    letter-spacing: 0.04em;
}

p, li, label, div[data-testid="stMarkdownContainer"], .stMarkdown, .stCaption {
    font-family: 'Share Tech Mono', monospace !important;
    color: #B8F5EA !important;
}

section[data-testid="stSidebar"] {
    background-color: #050A0E;
    border-right: 2px solid #00FFD1;
    box-shadow: inset -8px 0 20px -12px rgba(0,255,209,0.3);
}

div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00FFD1 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #00FFD1 !important;
    box-shadow: 0 0 6px #00FFD1;
}

div[data-testid="stCameraInput"], div[data-testid="stImage"] {
    border: 2px solid #00FFD1;
    box-shadow: 0 0 14px rgba(0,255,209,0.35), inset 0 0 14px rgba(0,255,209,0.08);
    padding: 6px;
    position: relative;
}

div[data-testid="stDataFrameResizable"], div[data-testid="stDataFrame"] {
    border: 1.5px solid #00FFD1 !important;
    box-shadow: 0 0 10px rgba(0,255,209,0.25);
}

div[data-testid="stAlert"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #0A1418 !important;
    border-left: 4px solid #00FFD1 !important;
    color: #B8F5EA !important;
}

hr {
    border-color: #00FFD1 !important;
    opacity: 0.4;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("Parámetros")
        st.subheader("Configuración de detección")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    picture = st.camera_input("Capturar imagen", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        # Decodificar con Pillow en lugar de cv2 (evita dependencia libGL)
        #pil_img  = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        #np_img   = np.array(pil_img)   # array RGB

        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR para que YOLO procese bien

        
        with st.spinner("Detectando objetos..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()              # devuelve BGR numpy array
        annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB sin cv2

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen con detecciones")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("Objetos detectados")
            if boxes is not None and len(boxes) > 0:
                label_names    = model.names
                category_count = {}
                category_conf  = {}

                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())
                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                data = [
                    {
                        "Categoría":          label_names[cat],
                        "Cantidad":           count,
                        "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                    }
                    for cat, count in category_count.items()
                ]

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")
                st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
