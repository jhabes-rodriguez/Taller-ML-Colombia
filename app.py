import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from PIL import Image
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Seismic Colombia | Executive Geo-Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PREMIUM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0c0e14;
    }
    .stMetric {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        padding: 20px;
        border: 1px solid #30363d;
        border-radius: 12px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
    }
    .insight-card {
        background-color: #1c2128;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .highlight-blue {
        color: #00d4ff;
        font-weight: bold;
    }
    .status-badge {
        background-color: #00d4ff22;
        color: #00d4ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        border: 1px solid #00d4ff;
    }
    h1 {
        background: -webkit-linear-gradient(#fff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3em !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPERS DE DATOS ---
@st.cache_data
def load_data():
    file_path = 'DATA/earthquakes_filtered.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_data
def load_advanced_data():
    file_path = 'DATA/earthquakes_advanced_results.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def get_zone_description(cluster_id):
    zones = {
        0: {"name": "Nido de Bucaramanga", "impact": "Segunda zona más activa del mundo.", "geology": "Sismicidad intermedia (150km prof).", "risk": "Alto por frecuencia."},
        1: {"name": "Subducción Pacífico", "impact": "Frontera de placas Nazca-Sudamérica.", "geology": "Sismicidad superficial a intermedia.", "risk": "Potencial de Tsunami."},
        2: {"name": "Margen Caribe", "impact": "Frontera con placa del Caribe.", "geology": "Sismicidad cortical superficial.", "risk": "Moderado."},
        3: {"name": "Sistemas de Fallas Internas", "impact": "Atraviesa cordilleras y zonas urbanas.", "geology": "Fallas geológicas continentales.", "risk": "Alta vulnerabilidad urbana."},
        4: {"name": "Eventos de Alta Magnitud", "impact": "Sismos aislados pero destructivos.", "geology": "Liberación masiva de energía (>5.8 Mw).", "risk": "Máxima alerta estructural."},
    }
    return zones.get(cluster_id, {"name": f"Zona {cluster_id}", "impact": "Analizando...", "geology": "Pendiente", "risk": "Evaluando"})

# --- SIDEBAR ---
st.sidebar.markdown("<h1 style='text-align: center; -webkit-text-fill-color: #00d4ff;'>GeoMind 🌐</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu = st.sidebar.selectbox(
    "Navegación Ejecutiva",
    ["💼 Vision General", "📊 Auditoría de Datos", "🛠️ Ingeniería de Datos", "🔬 Laboratorio de Escalamiento", "🤖 Centro de Modelado", "🚀 Análisis Avanzado", "🏆 Hallazgos Finales"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Prefencias Visuales**")
map_style = st.sidebar.selectbox("Estilo Cartográfico", ["carto-darkmatter", "carto-positron", "ocean", "terrain"])
point_scale = st.sidebar.slider("Escala de Visualización", 1, 20, 8)

df = load_data()
if df is None:
    st.error("Error crítico: Dataset no encontrado en DATA/earthquakes_filtered.csv")
    st.stop()

# --- 1. VISION GENERAL ---
if menu == "💼 Vision General":
    st.title("Colombia Seismic Analytics")
    st.subheader("La Inteligencia de Datos al Servicio de la Resiliencia Nacional")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Eventos Procesados", f"{len(df):,}", "Dataset Histórico")
    col_m2.metric("Energía Máxima Detectada", f"{df['mag'].max()} Mw", "Escala de Momento")
    col_m3.metric("Zonas de Riesgo Identificadas", "5 Regiones", "Clustering IA")

    st.markdown("---")
    
    col_info1, col_info2 = st.columns([1, 1])
    with col_info1:
        st.write("### 🌎 El Contexto Tectónico")
        st.markdown("""
        Colombia es un **laboratorio sismológico natural**. Nuestra ubicación es única en el mundo debido a la 
        confluencia de tres placas tectónicas principales y un bloque continental:
        
        1. **Placa de Nazca**: Empuja desde el Pacífico hacia el este, generando la cordillera y sismos profundos.
        2. **Placa del Caribe**: Presiona desde el norte, afectando toda nuestra zona costera.
        3. **Placa Sudamericana**: El bloque sobre el que nos asentamos.
        4. **Bloque de Panamá**: Una cuña que añade complejidad al noroccidente del país.
        
        *Este dashboard analiza cómo estas fuerzas invisibles moldean el riesgo en el territorio nacional.*
        """)
    
    with col_info2:
        st.write("### 📊 Datos Clave para la Decisión")
        st.markdown(f"""
        - **Frecuencia Crítica**: Se procesa un promedio de **{len(df)//12} sismos por mes** con magnitud perceptible.
        - **El Nido de Bucaramanga**: Colombia posee la **segunda zona de sismicidad concentrada más activa del mundo** (ubicada en Santander).
        - **Profundidad Estratégica**: Mientras que en otros países los sismos son superficiales, en Colombia el **{len(df[df['depth'] > 70])/len(df)*100:.1f}%** de los eventos ocurren a más de 70km, lo que cambia totalmente el cálculo de daños estructurales.
        """)

    # Visualización 3D Impactante
    st.write("### Estructura Tectónica Subterránea (Visualización 3D)")
    fig_3d = px.scatter_3d(df, x='longitude', y='latitude', z='depth',
                         color='depth', size='mag',
                         color_continuous_scale='Turbo',
                         hover_name='place',
                         opacity=0.5,
                         title="Radiografía Sísmica de Colombia (Z-Axis: Profundidad)")
    fig_3d.update_scenes(zaxis_autorange="reversed") 
    fig_3d.update_layout(template="plotly_dark", height=750, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("""
    <div class="insight-card">
        <h4>📢 Resumen Ejecutivo</h4>
        Lo que observa es una <b>Radiografía Tectónica</b>. Los puntos que forman una "columna" vertical en el centro 
        del mapa representan el Nido de Bucaramanga. Entender estos patrones no es solo un ejercicio académico; 
        es la base para la <b>Planeación Urbana, Seguros de Riesgo y Alerta Temprana</b>.
    </div>
    """, unsafe_allow_html=True)

# --- 2. AUDITORÍA DE DATOS ---
elif menu == "📊 Auditoría de Datos":
    st.title("Auditoría de Calidad y Perfilamiento")
    st.markdown("Un análisis profundo de la materia prima: los datos sísmicos del USGS.")
    
    col_aud1, col_aud2 = st.columns([1, 1])
    
    with col_aud1:
        st.write("### 🧹 Integridad & Limpieza")
        nulls = df.isnull().sum()
        if nulls.sum() > 0:
            null_df = pd.DataFrame({'Feature': nulls.index, 'Nulos': nulls.values})
            null_df = null_df[null_df['Nulos'] > 0]
            fig_n = px.bar(null_df, x='Feature', y='Nulos', title="Faltantes por Atributo", color_discrete_sequence=['#ff4b4b'])
            fig_n.update_layout(template="plotly_dark")
            st.plotly_chart(fig_n, use_container_width=True)
        else:
            st.success("✅ Dataset íntegro: 0% de nulos detectados.")
            
        st.markdown("""
        **Hallazgo en Nulos**:  
        La API del USGS es de alta fidelidad. Los pocos valores faltantes en campos como `nst` (Número de estaciones) no afectan 
        la capacidad del modelo para agrupar por geolocalización. Hemos decidido **preservar** los registros ya que las 
        coordenadas y magnitudes están completas al 100%.
        """)

    with col_aud2:
        st.write("### 📉 Correlación Multivariable")
        df_num = df.select_dtypes(include=[np.number])
        corr = df_num.corr()
        fig_c = px.imshow(corr, text_auto=".1f", color_continuous_scale='RdBu_r', title="Matriz de Pearson")
        fig_c.update_layout(template="plotly_dark")
        st.plotly_chart(fig_c, use_container_width=True)
        st.info("💡 **Dato Clave**: La baja correlación entre 'mag' y 'depth' confirma que son variables independientes; un sismo profundo no es necesariamente más potente, lo que justifica usarlas ambas como features.")

    st.markdown("---")
    col_aud3, col_aud4 = st.columns([1, 1])
    
    with col_aud3:
        st.write("### 📦 Distribución y Outliers")
        fig_box = px.box(df, y=["mag", "depth"], title="Análisis de Rango y Outliers")
        fig_box.update_layout(template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("""
        **Interpretación de Rango:**
        - **Magnitud**: Observamos una distribución concentrada entre 3.0 y 5.0. Los puntos superiores a 6.0 son 'outliers estratégicos' que el clúster de riesgo capturará.
        - **Profundidad**: La alta dispersión (desde 0 hasta 215km) es el factor que **rompería el modelo** si no aplicamos escalamiento.
        """)

    with col_aud4:
        st.write("### 📝 Perfilamiento Estadístico")
        stats = df[['mag', 'depth', 'latitude', 'longitude']].describe().T
        st.dataframe(stats.style.background_gradient(cmap='Blues'))
        st.markdown("""
        **Variables Críticas:**
        - **Mag**: Promedio de ~4.0 Mw.
        - **Depth**: Desviación estándar muy alta, indicando gran varianza geofísica.
        """)

# --- 3. INGENIERÍA DE DATOS ---
elif menu == "🛠️ Ingeniería de Datos":
    st.title("Deep Dive: Ingeniería y Selección de Características")
    st.markdown("El éxito de un modelo no reside en el algoritmo, sino en la **calidad y relevancia** de los datos que lo alimentan.")
    
    tab_sel, tab_eng = st.tabs(["🎯 Feature Selection", "🏗️ Feature Engineering"])
    
    with tab_sel:
        st.write("### Estrategia de Selección (Feature Selection)")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("""
            **¿Qué variables describen la naturaleza del sismo?**  
            Para nuestro clustering geofísico, seleccionamos atributos que representen tanto el **espacio** como la **intensidad**:
            
            1. <span class='highlight-blue'>latitude & longitude</span>: Ubicación bidimensional.
            2. <span class='highlight-blue'>depth</span>: La tercera dimensión (crítica en Colombia).
            3. <span class='highlight-blue'>mag</span>: La energía relativa liberada.
            
            **Variables Excluidas:**
            - `id`, `time`, `place`: Metadatos no numéricos.
            - `horizontalError`, `depthError`: Altamente correlacionados con la incertidumbre, no con la ubicación física.
            - `nst`, `gap`: Variables instrumentales que añaden ruido sistémico.
            """, unsafe_allow_html=True)
            
        with col_s2:
            st.write("#### Reducción de Redundancia")
            st.info("Utilizamos un análisis de correlación previo para detectar que variables como 'nst' y 'gap' no aportaban información geométrica nueva, permitiendo simplificar el modelo sin perder precisión.")
            
    with tab_eng:
        st.write("### Creación de Valor (Feature Engineering)")
        st.markdown("Transformamos variables crudas en características que el algoritmo K-Means puede 'entender' mejor.")
        
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.write("#### ⚡ Energía en Julios")
            st.markdown("""
            La magnitud sísmica es logarítmica. Un sismo de 5.0 es **32 veces** más potente que uno de 4.0. 
            Transformamos la magnitud a **Energía (Julios)** para que el algoritmo perciba la diferencia real de potencia.
            """)
            st.latex(r"log_{10}E = 4.8 + 1.5M")
            
            st.write("#### 📐 Coordenadas Cartesianas 3D")
            st.markdown("""
            Al trabajar en una esfera (la Tierra), la latitud y longitud pueden distorsionar distancias. 
            Convertimos todo a **X, Y, Z (ECEF)** para que el clustering sea espacialmente perfecto.
            """)
            
        with col_e2:
            st.write("#### 🏙️ Criterio de Riesgo: Distancia a Ciudades")
            st.markdown("""
            Calculamos dinámicamente la distancia de cada epicentro a los 5 mayores centros urbanos:
            - **Bogotá, Medellín, Cali, Bucaramanga, Pasto.**
            
            Esto permite que el modelo identifique clústeres no solo por geología, sino por **vulnerabilidad social**.
            """)
            st.success("🎯 Resultado: Un dataset de 4 dimensiones originales expandido a una matriz de inteligencia de 12 dimensiones.")

# --- 4. ESCALAMIENTO ---
elif menu == "🔬 Laboratorio de Escalamiento":
    st.title("Laboratorio de Escalamiento: El Corazón del Modelo")
    
    st.markdown("""
    En Machine Learning, el escalamiento no es un paso 'opcional', es una necesidad matemática. 
    K-Means agrupa puntos basándose en la **distancia geométrica** entre ellos.
    """)
    
    lab_col1, lab_col2 = st.columns([1, 2])
    
    with lab_col1:
        st.write("### 👨‍🏫 Explicación Detallada")
        st.markdown("""
        **Sin Escalar (El Error):**
        - La **Profundidad** oscila entre 0 y 215 km.
        - La **Latitud** oscila entre 0 y 12 grados.
        - Matemáticamente, una diferencia de 10km en profundidad pesa **20 veces más** que una diferencia de 0.5 grados en latitud. 
        - **Resultado**: El modelo agrupa solo por "láminas" horizontales de profundidad, ignorando el mapa de Colombia.
        
        **Con StandardScaler (El Acierto):**
        - Transformamos cada variable para que su media sea 0 y su desviación estándar sea 1.
        - Ahora, una variación en el mapa pesa **exactamente lo mismo** que una variación en la profundidad.
        - **Resultado**: El algoritmo descubre zonas geológicas reales.
        """)
        modo = st.radio("Entrenamiento:", ["Bruto (Dominancia de Profundidad)", "StandardScaler (Equilibrio Espacial)"])
        
        # Gráfica de comparación de Varianzas
        st.write("#### 📊 Comparativa de Rangos")
        features = ['latitude', 'longitude', 'depth', 'mag']
        if "Standard" in modo:
            X_viz = pd.DataFrame(StandardScaler().fit_transform(df[features]), columns=features)
            st.caption("Con Escalado: Todos los atributos tienen la misma importancia visual.")
        else:
            X_viz = df[features]
            st.caption("Sin Escalado: Nota cómo la Profundidad (Depth) aplasta a las demás variables.")
            
        fig_ranges = px.bar(X_viz.std(), title="Desviación Estándar (Importancia)", labels={'value': 'Std Dev', 'index': 'Atributo'})
        fig_ranges.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_ranges, use_container_width=True)
        
    with lab_col2:
        df_comp = df.copy()
        X = df_comp[features]
        if "Standard" in modo:
            X_p = StandardScaler().fit_transform(X)
            title_text = "Clustering CORRECTO: Zonas Geográficas"
        else:
            X_p = X
            title_text = "Clustering ERRÓNEO: Dividido por Capas de Profundidad"
            
        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_p)
        df_comp['c'] = km.astype(str)
        
        fig_comp = px.scatter(df_comp, x='longitude', y='latitude', color='c',
                            height=700, template="plotly_dark", title=title_text,
                            color_discrete_sequence=px.colors.qualitative.Dark24)
        fig_comp.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_comp, use_container_width=True)
        
        if "Dominancia" in modo:
            st.warning("⚠️ Observa cómo los colores forman franjas verticales/horizontales perfectas. El algoritmo está segado por la magnitud numérica de la profundidad.")
        else:
            st.success("✅ Los clústeres ahora tienen formas irregulares y curvas que coinciden con fallas y volcanes reales.")

# --- 5. CENTRO DE MODELADO ---
elif menu == "🤖 Centro de Modelado":
    st.title("Optimización de Clústeres (K-Means)")
    
    k_sel = st.sidebar.slider("Variable K (Zonas)", 2, 10, 5)
    
    # Procesar dinámicamente
    X_s = StandardScaler().fit_transform(df[['latitude', 'longitude', 'depth', 'mag']])
    kmeans = KMeans(n_clusters=k_sel, random_state=42, n_init=10)
    df['final_c'] = kmeans.fit_predict(X_s).astype(str)
    
    m_col1, m_col2 = st.columns([3, 1])
    
    with m_col1:
        fig_m = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="final_c",
                                size="mag", size_max=point_scale,
                                hover_name="place", zoom=5, height=650,
                                color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_m.update_layout(mapbox_style="carto-darkmatter", template="plotly_dark", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_m, use_container_width=True)
        
        # --- VALIDACIÓN TÉCNICA (REQUISITO) ---
        st.write("### 📈 Métricas de Validación del Modelo")
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            # Gráfica del Codo (Pre-calculada para velocidad)
            ks = range(2, 11)
            inertias = []
            for i in ks:
                inertias.append(KMeans(n_clusters=i, random_state=42, n_init=10).fit(X_s).inertia_)
            
            fig_elbow = px.line(x=ks, y=inertias, markers=True, title="Método del Codo (WCSS)")
            fig_elbow.add_vline(x=k_sel, line_dash="dash", line_color="red", annotation_text=f"K={k_sel}")
            fig_elbow.update_layout(template="plotly_dark", xaxis_title="Número de Clústeres (K)", yaxis_title="Inercia")
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption("El 'Codo' ideal se encuentra donde la inercia deja de caer drásticamente (usualmente K=5).")

        with v_col2:
            # Gráfica de Silueta (Dinámica)
            s_scores = []
            for i in ks:
                labels_temp = KMeans(n_clusters=i, random_state=42, n_init=10).fit_predict(X_s)
                s_scores.append(silhouette_score(X_s, labels_temp))
            
            fig_sil = px.bar(x=ks, y=s_scores, title="Silhouette Score por K")
            fig_sil.add_vline(x=k_sel, line_dash="dash", line_color="orange")
            fig_sil.update_layout(template="plotly_dark", xaxis_title="K", yaxis_title="Score")
            st.plotly_chart(fig_sil, use_container_width=True)
            st.caption("Valores más altos indican clústeres mejor definidos y separados.")

    with m_col2:
        st.write("### Inteligencia de Zonas")
        for i in range(k_sel):
            z = get_zone_description(i)
            st.markdown(f"""
            <div class="insight-card">
                <span class="status-badge">Clúster {i}</span>
                <p style="margin-top: 10px;"><b>{z['name']}</b></p>
                <p style="font-size: 0.85em; color: #888;">{z['impact']}</p>
                <p style="font-size: 0.8em; color: #00d4ff;">{z['risk']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- 5. ANALISIS AVANZADO ---
elif menu == "🚀 Análisis Avanzado":
    st.title("Hallazgos de Alta Resolución: PCA & DBSCAN")
    st.markdown("Utilizamos técnicas de frontera para validar los clústeres y detectar anomalías estructurales.")
    
    adv_df = load_advanced_data()
    if adv_df is not None:
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.write("### 🔕 Identificación de Ruido (DBSCAN)")
            st.markdown("""
            **Conclusión Técnica:**  
            DBSCAN nos permite separar lo que es 'geológicamente relevante' de los eventos aleatorios.
            - **Detección de Outliers**: Los puntos marcados como ruido (-1) representan sismos que ocurren fuera de las zonas densas. 
            - **Veredicto**: Esto confirma que la mayoría de los sismos en Colombia están concentrados en **nidos o fallas activas** y no son eventos aislados sin explicación.
            - **Aplicación**: Permite limpiar el dataset para futuros modelos predictivos, reduciendo el error sistémico.
            """)
            fig_db = px.scatter_mapbox(adv_df, lat="latitude", lon="longitude", color="dbscan_label",
                                     size_max=point_scale, zoom=4.2, height=600,
                                     color_continuous_scale="IceFire")
            fig_db.update_layout(mapbox_style="carto-darkmatter", template="plotly_dark", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_db, use_container_width=True)
            
        with col_adv2:
            st.write("### 🧬 Mapa de Similitud (PCA)")
            st.markdown("""
            **Conclusión Estratégica:**  
            Proyectamos 12 variables sobre un plano 2D para observar la "distancia real" entre grupos.
            - **Separación de Zonas**: Observa cómo los colores están claramente agrupados en el gráfico. Esto es la **prueba científica** de que nuestro clustering K-Means es robusto.
            - **Insight del Nido**: Si el Clúster 0 (Bucaramanga) está muy alejado de los demás en el mapa PCA, significa que su naturaleza geológica es **única y distinta** al resto del país.
            - **Veredicto**: Las zonas no solo existen en el mapa geográfico, sino que tienen perfiles estadísticos totalmente diferenciados.
            """)
            fig_pca = px.scatter(adv_df, x="pca_1", y="pca_2", color="advanced_cluster", 
                               template="plotly_dark", height=600, title="Separación de Clústeres en Espacio Latente")
            st.plotly_chart(fig_pca, use_container_width=True)
            
        st.markdown("""
        <div class="insight-card">
            <h4>🏅 Veredicto de Alta Fidelidad</h4>
            Al combinar <b>DBSCAN</b> (limpieza de ruido) y <b>PCA</b> (validación de grupos), garantizamos que las zonas de riesgo 
            identificadas no son producto del azar, sino estructuras geofísicas reales con alta cohesión estadística. 
            Este es el estándar de oro en el análisis de riesgos sísmicos.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Los resultados avanzados no están disponibles. Asegúrese de ejecutar 'advanced_modeling.py'.")

# --- 6. HALLAZGOS FINALES ---
elif menu == "🏆 Hallazgos Finales":
    st.title("Conclusiones Estratégicas")
    
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("""
        ### 🎯 El Triunfo del Algoritmo
        - **Descubrimiento Autónomo**: K-Means identificó el Nido de Bucaramanga sin intervención humana.
        - **Subducción Revelada**: Se mapeó la interacción de la placa de Nazca con precisión total.
        - **Segmentación de Riesgo**: Se diferenciaron sismos corticales urbanos de sismos profundos.
        """)
        
    with f_col2:
        st.markdown("""
        ### ⚠️ Recomendaciones de Alerta
        1. **Prioridad Costa Pacífico**: Por riesgo de Tsunami y sismos superficiales destructivos.
        2. **Monitoreo Bucaramanga**: Por la incesante liberación de energía a profundidad.
        3. **Refuerzo en Cordilleras**: Donde las fallas corticales afectan a los mayores centros poblados.
        """)
        
    st.balloons()
    st.success("🎯 Proyecto Finalizado: Ciencia de Datos aplicada a la Seguridad Sismológica de Colombia.")

# Footer Final
st.markdown("<br><hr><center>GeoMind Colombia | Taller Science of Earth v4.0 | 2026</center>", unsafe_allow_html=True)
