# TFG-Ocean-Debris-QG

Repositorio del Trabajo Fin de Grado de Ingeniería Aeroespacial sobre modelización y simulación del transporte y dispersión de restos de accidentes aéreos en el océano.

## Objetivo

Desarrollar una herramienta numérica en Python para simular una circulación oceánica idealizada y estudiar la dispersión lagrangiana de restos flotantes en un entorno geofísico simplificado.

## Alcance del proyecto

El trabajo combina:

- teoría de dinámica de fluidos geofísicos,
- aproximación cuasigeostrófica,
- ecuaciones de aguas someras,
- vorticidad potencial,
- transporte lagrangiano de partículas,
- análisis de resultados numéricos.

## Estructura del repositorio

- `src/`: código fuente principal.
- `data/`: datos de entrada y auxiliares.
- `figures/`: figuras generadas para la memoria.
- `results/`: resultados numéricos y salidas de simulación.
- `docs/`: documentación complementaria.
- `notebooks/`: cuadernos de exploración y análisis.
- `notes/`: apuntes internos, borradores y notas de trabajo.

## Requisitos

Instala las dependencias de Python con:

```bash
pip install -r requirements.txt
```

## Dependencias principales

- numpy
- scipy
- matplotlib
- pandas
- xarray
- netCDF4

## Estado actual

Estructura inicial del repositorio preparada para el desarrollo del TFG.

## Reproducibilidad

El proyecto está pensado para ser reproducible:

- el código principal vivirá en `src/`,
- los resultados se guardarán en `results/`,
- las figuras finales se exportarán a `figures/`,
- la documentación del proceso se mantendrá en `docs/` y `notes/`.

## Uso previsto

En fases posteriores del proyecto se incluirán:

- scripts de simulación,
- generación de partículas,
- análisis de trayectorias,
- visualización de resultados,
- documentación técnica de cada experimento.

## Licencia

Este repositorio incluye un archivo `LICENSE` en la raíz del proyecto.

## Autor

Guillermo Alba Buitrón
