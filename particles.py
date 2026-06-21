import numpy as np
from numba import njit

@njit(cache=True)
def _interp_bilinear(px: np.ndarray, py: np.ndarray, 
                     u_f: np.ndarray, v_f: np.ndarray,
                     dx: float, dy: float, nx: int, ny: int) -> tuple:
    """Interpolación bilineal de u y v - compilada con Numba."""
    n = px.shape[0]
    u_out = np.zeros(n)
    v_out = np.zeros(n)
    
    for i in range(n):
        fi = px[i] / dx
        fj = py[i] / dy
        i0 = int(np.floor(fi))
        j0 = int(np.floor(fj))
        
        i0 = max(0, min(i0, nx - 2))
        j0 = max(0, min(j0, ny - 2))
        
        i1 = i0 + 1
        j1 = j0 + 1
        wx = fi - i0
        wy = fj - j0
        
        u_out[i] = (u_f[i0, j0] * (1-wx)*(1-wy) + 
                  u_f[i1, j0] * wx*(1-wy) + 
                  u_f[i0, j1] * (1-wx)*wy + 
                  u_f[i1, j1] * wx*wy)
        
        v_out[i] = (v_f[i0, j0] * (1-wx)*(1-wy) + 
                   v_f[i1, j0] * wx*(1-wy) + 
                   v_f[i0, j1] * (1-wx)*wy + 
                   v_f[i1, j1] * wx*wy)
    
    return u_out, v_out

@njit(cache=True)
def _enforce_boundaries_numba(x: np.ndarray, y: np.ndarray, lx: float, ly: float):
    """Reflexión modular — compilada con Numba."""
    n = x.shape[0]
    for i in range(n):
        x[i] = _reflect_modular(x[i], lx)
        y[i] = _reflect_modular(y[i], ly)

@njit(cache=True)
def _reflect_modular(coord: float, limit: float) -> float:
    """Reflexión modular en [0, limit].
    
    Aplica rebote (bounce-back) para mantener la coordenada dentro del intervalo.
    No es una condición periódica (wrap-around), sino reflectante:
    - coord en [0, limit]  → se devuelve tal cual
    - coord en (limit, 2*limit] → se refleja: 2*limit - coord
    """
    c = coord % (2 * limit)
    if c <= limit:
        return c
    else:
        return 2 * limit - c

@njit(cache=True)
def _rk4_step_numba(x: np.ndarray, y: np.ndarray, 
                   u_f: np.ndarray, v_f: np.ndarray,
                   dt: float, dx: float, dy: float, 
                   nx: int, ny: int, lx: float, ly: float):
    """Avanza partículas un paso RK4 - compilado con Numba."""
    u1, v1 = _interp_bilinear(x, y, u_f, v_f, dx, dy, nx, ny)
    
    x2 = x + 0.5 * dt * u1
    y2 = y + 0.5 * dt * v1
    _enforce_boundaries_numba(x2, y2, lx, ly)
    u2, v2 = _interp_bilinear(x2, y2, u_f, v_f, dx, dy, nx, ny)
    
    x3 = x + 0.5 * dt * u2
    y3 = y + 0.5 * dt * v2
    _enforce_boundaries_numba(x3, y3, lx, ly)
    u3, v3 = _interp_bilinear(x3, y3, u_f, v_f, dx, dy, nx, ny)
    
    x4 = x + dt * u3
    y4 = y + dt * v3
    _enforce_boundaries_numba(x4, y4, lx, ly)
    u4, v4 = _interp_bilinear(x4, y4, u_f, v_f, dx, dy, nx, ny)
    
    x_new = x + (dt / 6.0) * (u1 + 2*u2 + 2*u3 + u4)
    y_new = y + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    
    _enforce_boundaries_numba(x_new, y_new, lx, ly)
    
    return x_new, y_new


class ParticleTracker:
    """Gestiona la advección de partículas pasivas en el flujo QG.
    
    La condición de contorno es reflectante (bounce-back), no periódica.
    Esto es coherente con una cuenca cerrada idealizada de doble giro:
    las partículas no pueden atravesar las fronteras sólidas (litoral),
    por lo que rebotan al alcanzar los límites del dominio [0, Lx] × [0, Ly].
    """
    
    def __init__(self, grid, n_particles: int, x0: float = None, y0: float = None,
                 radius: float = None, seed: int = None):
        self.grid = grid
        self.n = n_particles
        self.x, self.y = np.zeros(n_particles), np.zeros(n_particles)
        self.history = []
        self.rng = np.random.default_rng()
        if x0 is not None and y0 is not None and radius is not None:
            self._initialize_particles(x0, y0, radius, seed)

    def _initialize_particles(self, x0: float, y0: float, radius: float, seed: int = None):
        """Inicializa las partículas con distribución Gaussiana bidimensional (sigma = radius/2)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sigma = radius / 2.0
        self.x = x0 + sigma * self.rng.normal(size=self.n)
        self.y = y0 + sigma * self.rng.normal(size=self.n)
        _enforce_boundaries_numba(self.x, self.y, self.grid.lx, self.grid.ly)
        self.history = [np.column_stack((self.x, self.y))]

    def seed_cloud(self, x0: float, y0: float, radius: float, seed: int = None):
        """Libera una nube Gaussiana de partículas con sigma = radius / 2."""
        self._initialize_particles(x0, y0, radius, seed)

    def step(self, dt: float, u_field: np.ndarray, v_field: np.ndarray):
        """Avanza las partículas un paso dt usando RK4 lagrangiano (Numba compilado)."""
        self.x, self.y = _rk4_step_numba(
            self.x, self.y, u_field, v_field,
            dt, self.grid.dx, self.grid.dy,
            self.grid.nx, self.grid.ny, self.grid.lx, self.grid.ly
        )
        self.history.append(np.column_stack((self.x, self.y)))

    def save_trajectories(self, filename: str):
        """Exporta el historial completo de trayectorias."""
        np.savez(filename, trajectories=np.array(self.history))