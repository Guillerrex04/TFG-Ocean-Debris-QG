from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicsConfig:
    lx: float = 1.0e6      # Longitud zona E-W (m)
    ly: float = 1.0e6      # Longitud zona N-S (m)
    f0: float = 1.0e-4      # Parámetro de Coriolis central (s^-1)
    beta: float = 2.0e-11   # Gradiente de Coriolis (m^-1 s^-1)
    nu: float = 150.0       # Viscosidad horizontal (m^2/s)

@dataclass(frozen=True)
class NumericalConfig:
    nx: int = 129           # Nodos en x
    ny: int = 129           # Nodos en y
    dt: float = 3600.0      # Paso de tiempo (s)
