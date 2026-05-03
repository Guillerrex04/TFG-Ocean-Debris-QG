import numpy as np
from scipy.fft import dstn, idstn
from grid import Grid

class PoissonSolver:
    """
    Inversor de la ecuación de Poisson: laplacian(psi) = zeta.
    Antes utilizábamos scipy.sparse.linalg.spsolve (10-50× más rápido que SOR iterativo).
    Aun así, era lento. Ahora utiliza transformadas seno discretas (DST-I) para un dominio rectangular
    con condiciones de contorno Dirichlet homogéneas.
    """
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.nx, self.ny = grid.nx, grid.ny
        self.dx, self.dy = grid.dx, grid.dy
        self._nx_int = self.nx - 2
        self._ny_int = self.ny - 2
        self._eigenvalues = self._build_eigenvalues()

    def _build_eigenvalues(self):
        """Eigenvalores del laplaciano discreto para la base DST-I."""
        px = np.arange(1, self._nx_int + 1)[:, None]
        py = np.arange(1, self._ny_int + 1)[None, :]

        lam_x = -4.0 * np.sin(np.pi * px / (2.0 * (self._nx_int + 1)))**2 / (self.dx**2)
        lam_y = -4.0 * np.sin(np.pi * py / (2.0 * (self._ny_int + 1)))**2 / (self.dy**2)
        return lam_x + lam_y

    def solve(self, zeta: np.ndarray, psi_guess: np.ndarray = None) -> tuple:
        """
        Resuelve nabla^2(psi) = zeta con condición Dirichlet psi=0 en los bordes.
        Retorna (psi, info) donde info contiene métricas del solver.
        """
        interior = zeta[1:-1, 1:-1]

        # Transformada seno directa sobre el interior del dominio.
        zeta_hat = dstn(interior, type=1, norm='ortho')
        psi_hat = zeta_hat / self._eigenvalues
        psi_interior = idstn(psi_hat, type=1, norm='ortho')

        psi = np.zeros_like(zeta)
        psi[1:-1, 1:-1] = psi_interior
        psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1] = 0.0, 0.0, 0.0, 0.0

        info = {"solver": "dst_dirichlet"}
        return psi, info
