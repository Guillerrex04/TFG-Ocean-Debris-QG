import os
import sys
import warnings
import cupy as cp
from config import PhysicsConfig, NumericalConfig

DTYPE = cp.float64

class Grid:
    """Malla estructurada del dominio QG.
    
    Sistema de coordenadas:
        - x in [0, lx], y in [0, ly]  (físicas)
        - Origen: esquina suroeste (0, 0)
        - y positiva hacia el norte
        - Centro geométrico en (lx/2, ly/2)
        - Coordenadas centradas: xc = x - lx/2, yc = y - ly/2
    """
    def __init__(self, p: PhysicsConfig, n: NumericalConfig):
        self.nx, self.ny = n.nx, n.ny
        self.lx, self.ly = p.lx, p.ly
        self.dx = float(self.lx / (self.nx - 1))
        self.dy = float(self.ly / (self.ny - 1))
        
        self.x = cp.linspace(0, self.lx, self.nx, dtype=DTYPE)
        self.y = cp.linspace(0, self.ly, self.ny, dtype=DTYPE)
        self.X, self.Y = cp.meshgrid(self.x, self.y, indexing='ij')
        
        self.int = (slice(1, -1), slice(1, -1))
        self.shape = (self.nx, self.ny)

    def get_domain_center(self):
        """Devuelve (cx, cy) = (lx/2, ly/2), el centro geométrico del dominio."""
        return (self.lx / 2, self.ly / 2)

    def get_x_centered(self):
        """Coordenada x centrada: Xc = X - lx/2, origen en el centro del dominio."""
        return self.X - self.lx / 2

    def get_y_centered(self):
        """Coordenada y centrada: Yc = Y - ly/2, positiva hacia el norte."""
        return self.Y - self.ly / 2

    def get_mesh_centered(self):
        """Devuelve (Xc, Yc) — ambas mallas centradas en el centro del dominio."""
        return (self.get_x_centered(), self.get_y_centered())

    def get_y_normalized(self):
        """Coordenada y centrada y normalizada: (Y - ly/2) / (ly/2) en [-1, 1]."""
        return (self.Y - self.ly / 2) / (self.ly / 2)

    def get_beta_y(self, beta: float):
        """Término beta-planetario en coordenadas centradas: beta * Yc."""
        return beta * self.get_y_centered()

    def get_wind_stress_curl(self, p: PhysicsConfig):
        """Curl del esfuerzo del viento para doble giro (cuenca centrada).
        
        Crea un giro subpolar (curl negativo) y un giro subtropical (curl positivo),
        separados por la línea central de la cuenca. El argumento depende de la
        coordenada y centrada yc = y - ly/2.
        """
        yc = self.get_y_centered()
        curl_tau = p.tau0 * (2.0 * cp.pi / self.ly) * cp.sin(2.0 * cp.pi * yc / self.ly)
        return curl_tau
