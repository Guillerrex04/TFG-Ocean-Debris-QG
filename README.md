# TFG-Ocean-Debris-QG

Simulación QG (Quasi-Geostrophic) barotrópica oceánica para el estudio de dispersión de partículas.

**Trabajo de Fin de Grado — Guillermo Alba Buitrón**

## Estructura del proyecto

```
src/
  simulacion_oceano_qg_v1/   # Código fuente de la simulación
    config.py                 # Parámetros físicos y numéricos
    diagnostics.py            # Cálculo de métricas de dispersión
    grid.py                   # Generación de malla
    helpers.py                # Funciones auxiliares
    main.py                   # Punto de entrada: ejecuta la simulación
    operators.py              # Operadores en diferencias finitas
    particles.py              # Advección de partículas (RK4 + numba)
    plotting.py               # Gráficas de perfiles de viento
    poisson.py                # Solver de Poisson espectral (DST)
    post_process.py           # Post-procesado: vídeos, snapshots, gráficas
    qg_model.py               # Núcleo del modelo QG barotrópico
    test_numerical.py         # Tests de validación numérica
  animacion_conservacion_pv_columna/  # Animaciones adicionales
docs/                         # Documentación
notebooks/                    # Jupyter notebooks
data/                         # Datos de entrada
results/                      # Resultados numéricos
figures/                      # Figuras generadas
```

## Requisitos

- Python ≥ 3.9
- GPU NVIDIA con CUDA 12.x (para `cupy-cuda12x`)
- [uv](https://docs.astral.sh/uv/) (gestor de proyectos Python)

## Instalación

```bash
git clone https://github.com/Guillerrex04/TFG-Ocean-Debris-QG.git
cd TFG-Ocean-Debris-QG
uv sync
```

## Uso

Todos los comandos deben ejecutarse desde el directorio del código fuente:

```bash
cd src/simulacion_oceano_qg_v1
```

### Simulación

```bash
uv run python main.py
```

Ejecuta tres simulaciones consecutivas con N=10, N=100 y N=1000 partículas. Pregunta el día de liberación y guarda:
- Snapshots cada 0.5 días en `output/modelos_res_{n}/`
- Trayectorias en `output/netcdf/particles_output_{n}.nc`
- Métricas en `output/simulation_summary_res_{n}.npz`

### Post-procesado

```bash
uv run python post_process.py
```

Menú interactivo para generar:
- **Gráficas** (SVG): energía, enstrofía, área de búsqueda, dispersión RMS
- **Vídeos** (MP4): vorticidad, función de corriente, velocidad (con/sin partículas)
- **Snapshots** (PNG 2K): imágenes estáticas cada 50 días

### Tests numéricos

```bash
uv run python test_numerical.py
```

Valida el solucionador de Poisson y los operadores de velocidad.

## Flujo de trabajo típico

1. Ejecutar `uv run python main.py` para generar datos de simulación
2. Ejecutar `uv run python post_process.py` y seleccionar las visualizaciones deseadas

## Dependencias principales

- `numpy` — computación numérica
- `cupy-cuda12x` — aceleración GPU
- `scipy` — transformadas espectrales y geometría computacional
- `matplotlib` — visualización y animaciones
- `numba` — compilación JIT de advección de partículas
- `netCDF4` — almacenamiento de trayectorias
- `tqdm` — barras de progreso
