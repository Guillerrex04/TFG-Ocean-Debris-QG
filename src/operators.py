import numpy as np
from grid import Grid

class Operators:
    """Operadores de diferencias finitas para mallas rectangulares no periódicas."""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.dx = grid.dx
        self.dy = grid.dy

    def diff_x(self, f: np.ndarray) -> np.ndarray:
        """Derivada central df/dx calculada en el interior del dominio."""
        res = np.zeros_like(f)
        res[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * self.dx)
        return res

    def diff_y(self, f: np.ndarray) -> np.ndarray:
        """Derivada central df/dy calculada en el interior del dominio."""
        res = np.zeros_like(f)
        res[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * self.dy)
        return res

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Laplaciano estándar de 5 puntos (2º orden) en el interior."""
        res = np.zeros_like(f)
        idx = self.grid.int
        res[idx] = (
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dx**2 +
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dy**2
        )
        return res

    def arakawa_jacobian(self, psi: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calcula el Jacobiano de Arakawa J(psi, q).
        
        Este esquema de 9 puntos mejora la conservación discreta de energía y enstrofía 
        y reduce errores no lineales espurios frente a un jacobiano centrado simple.
        
        Ref: Arakawa (1966), J. Comput. Phys. 1, 119-143.
        """
        j = np.zeros_like(psi)
        dx, dy = self.dx, self.dy
        
        # Combinación de las tres formas discretas equivalentes del Jacobiano
        j_pp = ( (psi[2:, 1:-1] - psi[:-2, 1:-1]) * (q[1:-1, 2:] - q[1:-1, :-2]) -
                 (psi[1:-1, 2:] - psi[1:-1, :-2]) * (q[2:, 1:-1] - q[:-2, 1:-1]) )
        
        j_px = ( psi[2:, 2:]   * (q[1:-1, 2:] - q[2:, 1:-1]) -
                 psi[:-2, :-2] * (q[:-2, 1:-1] - q[1:-1, :-2]) -
                 psi[:-2, 2:]  * (q[1:-1, 2:] - q[:-2, 1:-1]) +
                 psi[2:, :-2]  * (q[2:, 1:-1] - q[1:-1, :-2]) )
        
        j_xp = ( q[2:, 2:]   * (psi[2:, 1:-1] - psi[1:-1, 2:]) -
                 q[:-2, :-2] * (psi[1:-1, :-2] - psi[:-2, 1:-1]) -
                 q[:-2, 2:]  * (psi[:-2, 1:-1] - psi[1:-1, 2:]) +
                 q[2:, :-2]  * (psi[1:-1, :-2] - psi[2:, 1:-1]) )

        j[1:-1, 1:-1] = (j_pp + j_px + j_xp) / (12.0 * dx * dy)
        return j
