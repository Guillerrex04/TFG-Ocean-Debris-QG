from dataclasses import dataclass, field
import numpy as np

DTYPE = "float64"

@dataclass(frozen=True)
class PhysicsConfig:
    # Parámetros geométricos y físicos configurables
    lx: float = 6.0e6           # Ancho del dominio [m] (6000 km, dirección zonal)
    ly: float = 3.0e6           # Alto del dominio [m] (3000 km, dirección meridional)
    latitud_deg: float = 40     # Latitud de referencia [grados] (+Norte, -Sur)
    nu: float = 10              # Coeficiente de viscosidad lateral [m²/s]
    H: float = 1000.0           # Profundidad de la capa oceánica [m]
    rho0: float = 1025.0        # Densidad de referencia agua de mar [kg/m³]
    tau0: float = 0.2           # Amplitud máxima esfuerzo del viento [N/m²] (Pa)
    r: float = 1e-7             # Coeficiente de fricción de fondo [1/s]

    # Parámetros de Coriolis calculados dinámicamente
    f0: float = field(init=False)     # Parámetro de Coriolis de referencia [1/s]
    beta: float = field(init=False)   # Variación meridional de Coriolis [1/(m·s)]

    def __post_init__(self):
        omega = 7.2921e-5       # Velocidad angular de la Tierra [rad/s]
        R_tierra = 6.371e6      # Radio medio de la Tierra [m]
        phi = np.deg2rad(self.latitud_deg)
        f0_calc = 2.0 * omega * np.sin(phi)
        beta_calc = (2.0 * omega * np.cos(phi)) / R_tierra
        object.__setattr__(self, 'f0', f0_calc)
        object.__setattr__(self, 'beta', beta_calc)

@dataclass(frozen=True)
class NumericalConfig:
    nx: int = 513                  # Número de puntos de malla en dirección X [-]
    ny: int = 257                  # Número de puntos de malla en dirección Y [-]
    # Con nx=513, lx=6e6  → dx = 6e6 / (513-1) = 11718.75 m (~11.72 km)
    # Con ny=257, ly=3e6  → dy = 3e6 / (257-1) = 11718.75 m (~11.72 km)
    # Dominio: 6000 km × 3000 km, resolución ~11.71 km (isotrópica)
    dt: float = 3600.0             # Paso de tiempo de la simulación [s] (1 hora)
    cfl_safety: float = 0.8        # Factor de seguridad para condición CFL [-]
    save_interval_days: int = 1    # Intervalo de guardado [días]
    n_particles: int = 500         # Número de partículas para dispersión lagrangiana [-]
    release_radius: float = 2000.0 # Radio inicial de la nube de partículas [m]