import numpy as np
from config import PhysicsConfig, NumericalConfig

class Grid:
    """Geometría y utilidades de acceso a la malla."""
    def __init__(self, p: PhysicsConfig, n: NumericalConfig):
        self.nx, self.ny = n.nx, n.ny
        self.lx, self.ly = p.lx, p.ly
        
        self.x = np.linspace(0, self.lx, self.nx)
        self.y = np.linspace(0, self.ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)
        
        # Helper para indexado del interior (excluye bordes)
        self.int = (slice(1, -1), slice(1, -1))
        self.shape = (self.nx, self.ny)

    def get_beta_y(self, beta: float):
        """Retorna el término beta*y centrado en el dominio."""
        return beta * (self.Y - self.ly / 2)
