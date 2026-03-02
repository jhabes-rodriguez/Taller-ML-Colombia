# Reporte Ejecutivo: Clustering de Actividad Sísmica en Colombia

## Resumen
**Pregunta de investigación:** ¿Es posible identificar automáticamente zonas sísmicas diferenciadas en Colombia utilizando únicamente las características instrumentales de los sismos?

Colombia es uno de los países con mayor actividad sísmica del planeta. Ubicado en la intersección de tres placas tectónicas — Nazca, Caribe y Sudamericana — el territorio colombiano experimenta miles de sismos cada año, la mayoría imperceptibles pero algunos devastadores.

El problema abordado es la identificación de patrones espaciales y de magnitud en esta sismicidad histórica para revelar estructuras sismogénicas sin depender de mapas geológicos previos.

## Contexto del Proyecto (Business Understanding)
- **Definición del Problema**: Falta de una segmentación automática de la actividad sísmica basada en datos instrumentales para descubrir estructuras ocultas y agrupaciones naturales.
- **Justificación de Clustering**: Al no tener etiquetas predefinidas, el aprendizaje no supervisado permite que los datos revelen la "anatomía" de la sismicidad sin sesgos.
- **Stakeholders**: Servicio Geológico Colombiano (SGC), UNGRD (Gestión del Riesgo), urbanistas y aseguradoras.
- **Hipótesis**: Se espera identificar clústeres que reflejen la interacción de las tres placas tectónicas principales y fenómenos locales de alta densidad como los nidos sísmicos.

## Metodología (CRISP-DM)
- **Comprensión de datos**: Extracción de datos de la API de USGS y análisis exploratorio inicial.
- **Preparación**: Limpieza de nulos y selección de variables (`latitude`, `longitude`, `depth`, `mag`).
- **Modelado**: Aplicación de K-Means y técnica del "Codo" para hallar el número óptimo de clústeres (k).
- **Evaluación**: Análisis de la cohesión de los grupos y relevancia geológica.

## Resultados Preliminares (Análisis Exploratorio)
- **Conteo de Datos**: Se procesaron **6,854 registros** totales, de los cuales **1,412** están ubicados específicamente en territorio colombiano.
- **Calidad de Variables**: Las dimensiones clave para el clustering (`mag`, `depth`, `latitude`, `longitude`) presentan un **0% de valores nulos**, garantizando la robustez del modelo.
- **Distribución de Magnitud y Profundidad**:
![Distribuciones](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\distributions.png)

- **Correlaciones**: Se identificó una fuerte correlación entre la ubicación y la profundidad en ciertas zonas.
![Mapa de Calor de Correlación](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\correlation_heatmap.png)

### Visualización Geográfica (EDA)
![Mapa por Profundidad](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\map_depth.png)
*Mapa coloreado por profundidad, destacando el Nido de Bucaramanga.*

![Mapa por Magnitud](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\map_magnitude.png)
*Mapa coloreado por magnitud.*

## Fase 4: Modelamiento

### 4.1 Método del Codo y Silhouette Score
Se ejecutó el algoritmo K-Means para un rango de $k$ de 2 a 10 para identificar el punto óptimo de segmentación.

![Gráficos de Métricas](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\elbow_method.png)
![Análisis de Silueta](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\silhouette_score.png)

### 4.2 Selección de K (Justificación)
Se ha seleccionado **K=5** como el valor óptimo. La justificación se basa en tres pilares:

1.  **Métrica de Inercia (Codo)**: El gráfico de WCSS muestra una disminución pronunciada hasta K=4 y un "suavizado" claro en K=5, indicando que añadir más clústeres empieza a aportar rendimientos decrecientes en la reducción de la varianza interna.
2.  **Métrica de Silueta**: El Silhouette Score para K=5 es de **0.4541**, uno de los valores más altos y estables. Aunque K=7 tiene un score ligeramente superior (0.48), la ganancia estadística no justifica la complejidad adicional de manejar 7 zonas diferenciadas.
3.  **Sentido Geológico**: K=5 permite aislar perfectamente el **Nido Sísmico de Bucaramanga** y diferenciar las zonas de subducción (Pacífico) de las fallas continentales (Andina/Caribe), proporcionando la segmentación más útil para el SGC sin sobre-segmentar los datos.

## Fase 5: Evaluación (Perfiles de Clústeres)

A continuación se detalla la anatomía de cada uno de los 5 clústeres identificados, proporcionando una base para la toma de decisiones del SGC:

### 5.1 Perfil Estadístico e Interpretación

1.  **Clúster 0: Nido Sísmico de Bucaramanga (Santander)**
    - **Sismos**: 714 registros (El más denso).
    - **Profundidad**: Media de **152.7 km** (Desv: 15.8 km).
    - **Interpretación**: Representa uno de los fenómenos sísmicos más activos del mundo. Es una zona de sismicidad intermedia concentrada en un volumen muy pequeño.
2.  **Clúster 1: Eventos de Alta Magnitud (Sismicidad Crítica)**
    - **Sismos**: 100 registros.
    - **Magnitud**: Media de **5.27** (Máx: 7.3). Es el grupo con mayor liberación de energía.
    - **Interpretación**: Sismos dispersos pero potentes que requieren monitoreo estructural prioritario.
3.  **Clúster 2: Zona Pacífico y Subducción Superficial**
    - **Sismos**: 329 registros.
    - **Geografía**: Occidente colombiano (Costa Pacífica).
    - **Interpretación**: Sismos superficiales (~42 km) generados por la interacción directa de la Placa de Nazca al entrar bajo la placa Sudamericana.
4.  **Clúster 3: Cordilleras y Fallas Continentales Centrales**
    - **Sismos**: 164 registros.
    - **Profundidad**: Muy superficial (Media: **23.8 km**).
    - **Interpretación**: Sismicidad cortical generada por los sistemas de fallas dinámicos que atraviesan el corazón de las cordilleras colombianas.
5.  **Clúster 4: Margen Caribe y Norte de Colombia**
    - **Sismos**: 105 registros.
    - **Geografía**: Norte y Caribe colombiano.
    - **Interpretación**: Representa la interacción sismogénica de la Placa del Caribe con el norte del país, caracterizada por ser superficial y geográficamente extensa.

### 5.3 Hallazgos Finales

-   **¿Hay algún clúster de sismicidad profunda concentrada?**
    Sí, el **Clúster 0**. Coincide exactamente con el **Nido Sísmico de Bucaramanga**. Es impresionante cómo el algoritmo aisló esta columna de sismos que promedian los 150 km de profundidad en una ubicación geográfica muy compacta en Santander.
-   **¿Qué clústeres coinciden con la zona de subducción del Pacífico?**
    El **Clúster 2**. Sus coordenadas y su profundidad media (~42 km) reflejan perfectamente la zona donde la placa de Nazca se introduce bajo la Sudamericana a lo largo del litoral pacífico.
-   **¿Algún clúster captura los sismos de mayor magnitud?**
    El **Clúster 1**. Es el grupo que captura los sismos con magnitudes superiores (Media: 5.27, Máx: 7.3). Aunque son eventos menos frecuentes que los del Nido de Bucaramanga, representan la mayor amenaza por su liberación de energía.
-   **¿Qué clúster recomendarías priorizar para alertas tempranas y por qué?**
    Se recomienda priorizar el **Clúster 1 (Alta Magnitud)** y el **Clúster 2 (Pacífico)**. 
    - El Clúster 1 por la obvia peligrosidad de su energía (magnitud). 
    - El Clúster 2 porque, al ser sismos superficiales y ocurrir cerca de la costa, tienen un alto potencial de generar tsunamis y daños estructurales inmediatos en centros poblados.

> [!TIP]
> **Conclusiones del Taller**: Al investigar el "Nido de Bucaramanga", confirmamos que nuestro modelo de K-Means (un algoritmo de 1957) fue capaz de detectar automáticamente uno de los tres nidos sísmicos más importantes del mundo, validando la potencia del aprendizaje no supervisado cuando los datos están correctamente preparados y escalados.

- **Fenómeno descubierto**: El algoritmo identificó automáticamente la diferencia entre sismos de falla superficial (Caribe/Pacífico) y sismos de subducción profunda (Andinos), logrando aislar el nido sísmico de Bucaramanga como una entidad estadística independiente.


## Fase 3: Preparación de Datos

### 3.1 Selección de Features
Para garantizar un clustering robusto y evitar distorsiones por datos faltantes, se seleccionaron las siguientes variables numéricas:

1.  **Variables de Ubicación**: `latitude`, `longitude` y `depth`. Estas definen la posición tridimensional del evento.
2.  **Variable de Naturaleza**: `mag` (magnitud), que describe la energía liberada.
3.  **Justificación de Exclusión**: Variables como `nst` (60.4% nulos), `horizontalError` (20.3% nulos) y `magError` (15.8% nulos) fueron descartadas debido a su baja integridad en el subset colombiano, lo que afectaría la precisión de las distancias en K-Means.

### 3.2 Manejo de datos faltantes
- **Estrategia**: Se optó por **seleccionar únicamente features con 0% de nulos** (`latitude`, `longitude`, `depth`, `mag`).
- **Decisión**: Dado que el objetivo es geográfico y estructural, eliminar registros (filas) con nulos en variables secundarias reduciría el dataset innecesariamente. Tampoco se utilizó imputación para evitar introducir ruido estadístico en variables con alta ausencia de datos (como `nst`).

## Fase 3.3: Scaling (CRÍTICO)

A continuación, se presenta la comparación entre aplicar K-Means con y sin escalamiento:

![Comparativa de Escalamiento](C:\Users\IVONNE ENRIQUEZ\.gemini\antigravity\brain\3b3ad3f2-94be-42c8-bed3-41b41fbc78b0\scaling_comparison.png)

### Análisis del Experimento
1.  **¿Cambian los clústeres al escalar? ¿Por qué?**
    Sí, radicalmente. Sin escalar, la **profundidad** (`depth`) domina la distancia euclidiana debido a su magnitud. Al escalar con `StandardScaler`, todas las variables (latitud, longitud, profundidad, magnitud) tienen el mismo peso estadístico, permitiendo descubrir patrones geológicos ocultos.
2.  **¿Cuál es la escala de latitude vs depth? ¿Qué feature domina si no escalas?**
    - **Latitud**: Rango de ~13 unidades (grados).
    - **Profundidad**: Rango de **~215 unidades** (km).
    - **Dominancia**: Sin escalamiento, la profundidad domina el modelo al ser numéricamente ~16 veces mayor.
3.  **¿Cuál versión produce clústeres más interpretables para el SGC?**
    La versión **con escalamiento**. Permite identificar zonas que geográficamente se superponen (mismas coordenadas x e y) pero que geológicamente son distintas por su profundidad (ej. fallas de placa superficiales frente a subducción profunda).

---

## Impacto del Scaling
Como se demostró, sin el paso de escalamiento, el algoritmo ignoraría la ubicación geográfica fina a favor de la profundidad masiva. Con el escalamiento, todas las variables tienen media 0 y desviación estándar 1, permitiendo una comparación equitativa.




## Recomendaciones para el SGC
- **Monitoreo Prioritario**: Se recomienda priorizar el Clúster 1 (Pacífico) y el Clúster 3 (Caribe) para estudios de riesgo civil, ya que aunque sus magnitudes son similares a otros clústeres, su baja profundidad aumenta el potencial de daños en infraestructura.
- **Microzonificación**: Utilizar los centros de los clústeres para afinar la ubicación de nuevas estaciones de monitoreo sísmico.


## Conclusiones
- **K-Means puede**: Agrupar eficientemente grandes volúmenes de datos y encontrar centros geográficos de actividad sismogénica diferenciando subducción de fallas superficiales.
- **K-Means NO puede**: Predecir cuándo ocurrirá el próximo sismo ni manejar formas de clústeres no esféricas (como fallas geológicas lineales que recorren cientos de kilómetros).

