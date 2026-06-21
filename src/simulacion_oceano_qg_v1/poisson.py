import warnings
import cupy as cp
from grid import Grid

DTYPE = cp.float64
_TWO = cp.float64(2.0)


def _dst_forward(f: cp.ndarray) -> cp.ndarray:
    ni, nj = f.shape
    nx_ext = 2 * (ni + 1)
    ny_ext = 2 * (nj + 1)

    ext = cp.zeros((nx_ext, ny_ext), dtype=DTYPE)
    ext[1:ni+1, 1:nj+1] = f
    ext[nx_ext-1:ni+1:-1, 1:nj+1] = -f
    ext[1:ni+1, ny_ext-1:nj+1:-1] = -f
    ext[nx_ext-1:ni+1:-1, ny_ext-1:nj+1:-1] = f

    F = cp.fft.fft2(ext)
    N, M = float(ni + 1), float(nj + 1)
    return F[1:ni+1, 1:nj+1].real * (_TWO / (N * M))


def _dst_inverse(F_hat: cp.ndarray) -> cp.ndarray:
    ni, nj = F_hat.shape
    nx_ext = 2 * (ni + 1)
    ny_ext = 2 * (nj + 1)

    ext = cp.zeros((nx_ext, ny_ext), dtype=DTYPE)
    ext[1:ni+1, 1:nj+1] = F_hat
    ext[nx_ext-1:ni+1:-1, 1:nj+1] = -F_hat
    ext[1:ni+1, ny_ext-1:nj+1:-1] = -F_hat
    ext[nx_ext-1:ni+1:-1, ny_ext-1:nj+1:-1] = F_hat

    f = cp.fft.ifft2(ext)
    N, M = float(ni + 1), float(nj + 1)
    return f[1:ni+1, 1:nj+1].real * (N * M / _TWO)


class PoissonSolver:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.nx, self.ny = grid.nx, grid.ny
        self.dx = float(grid.dx)
        self.dy = float(grid.dy)
        self._nx_int = self.nx - 2
        self._ny_int = self.ny - 2
        self._eigenvalues = self._build_eigenvalues()

    def _build_eigenvalues(self):
        nx = self._nx_int
        ny = self._ny_int
        k_idx = cp.arange(1, nx + 1, dtype=DTYPE)[:, None]
        l_idx = cp.arange(1, ny + 1, dtype=DTYPE)[None, :]

        kx = k_idx * cp.pi / float(nx + 1)
        ky = l_idx * cp.pi / float(ny + 1)

        ev = (-_TWO * cp.cos(kx) + _TWO) / (self.dx ** 2) + \
             (-_TWO * cp.cos(ky) + _TWO) / (self.dy ** 2)
        return ev.astype(DTYPE)

    def solve(self, zeta: cp.ndarray, psi_guess: cp.ndarray = None):
        interior = zeta[1:-1, 1:-1].astype(DTYPE)

        zeta_hat = _dst_forward(interior)
        psi_hat = -zeta_hat / self._eigenvalues
        psi_interior = _dst_inverse(psi_hat)

        psi = cp.zeros_like(zeta, dtype=DTYPE)
        psi[1:-1, 1:-1] = psi_interior

        return psi, {"solver": "dst1_fft"}