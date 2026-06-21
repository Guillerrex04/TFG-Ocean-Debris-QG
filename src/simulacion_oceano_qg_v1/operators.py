import cupy as cp
import numpy as np
from grid import Grid

DTYPE = cp.float64

_TWO = cp.float64(2.0)
_FOUR = cp.float64(4.0)
_TWELVE = cp.float64(12.0)
_HALF = cp.float64(0.25)


class Operators:
    """Operadores de diferencias finitas - Optimizado float64."""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.dx = cp.float64(grid.dx)
        self.dy = cp.float64(grid.dy)

    def diff_x(self, f: cp.ndarray) -> cp.ndarray:
        res = cp.zeros_like(f, dtype=DTYPE)
        res[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (_TWO * self.dx)
        return res

    def diff_y(self, f: cp.ndarray) -> cp.ndarray:
        res = cp.zeros_like(f, dtype=DTYPE)
        res[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (_TWO * self.dy)
        return res

    def laplacian(self, f: cp.ndarray) -> cp.ndarray:
        res = cp.zeros_like(f, dtype=DTYPE)
        idx = self.grid.int
        res[idx] = (
            (f[2:, 1:-1] - _TWO * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (self.dx ** 2) +
            (f[1:-1, 2:] - _TWO * f[1:-1, 1:-1] + f[1:-1, :-2]) / (self.dy ** 2)
        )
        return res

    def arakawa_jacobian(self, psi: cp.ndarray, q: cp.ndarray) -> cp.ndarray:
        j = cp.zeros_like(psi, dtype=DTYPE)
        dx, dy = self.dx, self.dy
        
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

        j[1:-1, 1:-1] = (j_pp + j_px + j_xp) / (_TWELVE * dx * dy)
        return j