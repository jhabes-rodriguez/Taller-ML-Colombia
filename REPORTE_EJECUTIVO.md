# Reporte Ejecutivo: Clustering de Actividad Sísmica en Colombia

## Resumen
Este análisis aborda la complejidad tectónica de Colombia mediante el uso de algoritmos de **Aprendizaje No Supervisado (K-Means)** para segmentar zonas de riesgo sísmico. El problema central radica en la identificación de patrones en un entorno donde convergen múltiples placas tectónicas y bloques continentales. 

Utilizando un dataset histórico del USGS, aplicamos un flujo de trabajo basado en **CRISP-DM**, integrando ingeniería de datos avanzada para mejorar la precisión espacial. El hallazgo principal fue el redescubrimiento autónomo del **Nido de Bucaramanga** (una de las zonas más activas del mundo) y la delimitación de la zona de subducción del Pacífico, validando que el Machine Learning puede decodificar estructuras geológicas profundas sin supervisión humana previa.

## Metodología
El proyecto siguió las fases de la metodología **CRISP-DM**:
1. **Business Understanding**: Identificación de la necesidad de segmentar zonas sísmicas para la prevención de desastres.
2. **Data Understanding (EDA)**: Análisis de 3,514 eventos, revelando distribuciones multimodales en profundidad y una alta concentración en Santander.
3. **Data Preparation**: Tratamiento de nulos, eliminación de redundancia (Feature Selection) y aplicación crítica de **StandardScaler** para equilibrar magnitudes de profundidad (0-215 km) con coordenadas (0-12°).
4. **Modeling**: Ejecución de K-Means (K=5 optimizado por el método del Codo y Silueta) y modelos avanzados (DBSCAN/PCA).
5. **Evaluation**: Validación geológica de los clústeres y creación de un dashboard interactivo en Streamlit.
6. **Deployment**: Entrega del sistema de visualización ejecutivo.

## Resultados
Se identificaron 5 zonas clave (K=5):
- **Clúster 0 (Nido de Bucaramanga)**: Sismicidad intermedia (profundidad ~150km). Alta frecuencia pero bajo impacto superficial directo.
- **Clúster 1 (Pacífico/Subducción)**: Sismos superficiales en el litoral pacífico. Potencial de riesgo por tsunami.
- **Clúster 2 (Zona Norte)**: Actividad moderada en el margen del Caribe.
- **Clúster 3 (Sistemas Andinos)**: Sismicidad cortical que atraviesa centros urbanos (Bogotá, Medellín). Alta vulnerabilidad.
- **Clúster 4 (Eventos Críticos)**: Sismos de alta magnitud (>5.5 Mw) distribuidos en fallas principales.

**Hallazgo Maestro**: El algoritmo aisló el Nido de Bucaramanga como un clúster independiente debido a su perfil único de "profundidad vs frecuencia", lo que demuestra la robustez del modelo.

## Impacto del Scaling
El escalamiento fue el factor determinante del éxito. 
- **Sin Escalar**: El K-Means se vio sesgado por la variable `depth` (debido a su rango numérico dominante), agrupando los sismos en "capas horizontales" que ignoraban la geografía de Colombia.
- **Con Scaling (StandardScaler)**: Todas las variables (Lat, Lon, Depth, Mag) compartieron la misma "voz" matemática. Esto permitió que los clústeres adoptaran formas geológicas reales, respetando tanto la ubicación en el mapa como el comportamiento vertical.
- **Lección**: En geofísica, los atributos tienen unidades y rangos incompatibles; el escalamiento es el traductor universal que permite el descubrimiento de patrones.

## Recomendaciones para el SGC (Servicio Geológico Colombiano)
1. **Priorización de Monitoreo**: Reforzar la red de sensores en el **Clúster 1 (Pacífico)** debido a la naturaleza superficial de sus sismos y el riesgo de tsunami para poblaciones costeras.
2. **Microzonificación Urbana**: Centrar estudios detallados en el **Clúster 3**, ya que mapea las fallas que afectan directamente a las metrópolis andinas.
3. **Inteligencia de Datos**: Integrar modelos de clustering dinámico en tiempo real para detectar cambios en la "densidad sísmica", lo que podría indicar precursores de eventos mayores en el Nido de Bucaramanga.

## Conclusiones
El uso de K-Means demostró ser altamente efectivo para la segmentación espacial y el descubrimiento de estructuras geológicas a gran escala. 
- **Fortaleza**: Capacidad de procesar miles de eventos y encontrar grupos que coinciden con la realidad tectónica nacional. 
- **Limitación**: K-Means asume formas esféricas y no maneja bien el "ruido" (sismos aislados). Por ello, el uso complementario de **DBSCAN** fue vital para identificar eventos atípicos. 
- **Reflexión**: El Machine Learning no reemplaza al geólogo, pero le proporciona un "ojo biónico" para ver patrones invisibles en el Big Data sísmico.

## Referencias
- **USGS Earthquake Hazards Program**: Fuente principal de datos.
- **Servicio Geológico Colombiano (SGC)**: Guía de zonificación sísmica nacional.
- **Scikit-learn Documentation**: Referencia técnica para K-Means, DBSCAN y PCA.
- **Taller Machine Learning**: Notebooks de EDA y Modelado desarrollados en el taller.
