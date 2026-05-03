import numpy as np
from grid import Grid

class ParticleTracker:
    """Gestiona la advección de partículas pasivas en el flujo QG."""
    
    def __init__(self, grid: Grid, n_particles: int):
        self.grid = grid
        self.n = n_particles
        self.x, self.y = np.zeros(n_particles), np.zeros(n_particles)
        self.history = []
        self.rng = np.random.default_rng()

    def seed_cloud(self, x0: float, y0: float, radius: float, seed: int = None):
        """Libera una nube circular de partículas con generador local."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = radius * np.sqrt(self.rng.random(self.n))
        theta = 2 * np.pi * self.rng.random(self.n)
        self.x, self.y = x0 + r * np.cos(theta), y0 + r * np.sin(theta)
        self._enforce_boundaries()
        self.history = [np.column_stack((self.x, self.y))]

    def _enforce_boundaries(self):
        """Reflexión modular robusta en las paredes [0, L]."""
        def reflect(coord, limit):
            c = coord % (2 * limit)
            return np.where(c <= limit, c, 2 * limit - c)
        self.x, self.y = reflect(self.x, self.grid.lx), reflect(self.y, self.grid.ly)

    def _get_velocity_at(self, px, py, u_f, v_f):
        """Interpolación bilineal de u y v."""
        fi, fj = px / self.grid.dx, py / self.grid.dy
        i0, j0 = np.floor(fi).astype(int).clip(0, self.grid.nx - 2), np.floor(fj).astype(int).clip(0, self.grid.ny - 2)
        i1, j1, wx, wy = i0 + 1, j0 + 1, fi - i0, fj - j0
        def interp(f):
            return (f[i0, j0] * (1-wx)*(1-wy) + f[i1, j0] * wx*(1-wy) +
                    f[i0, j1] * (1-wx)*wy     + f[i1, j1] * wx*wy)
        return interp(u_f), interp(v_f)

    def step(self, dt: float, u_field: np.ndarray, v_field: np.ndarray):
        """Avanza las partículas un paso dt usando RK4 lagrangiano."""
        u1, v1 = self._get_velocity_at(self.x, self.y, u_field, v_field)
        u2, v2 = self._get_velocity_at(self.x + 0.5*dt*u1, self.y + 0.5*dt*v1, u_field, v_field)
        u3, v3 = self._get_velocity_at(self.x + 0.5*dt*u2, self.y + 0.5*dt*v2, u_field, v_field)
        u4, v4 = self._get_velocity_at(self.x + dt*u3, self.y + dt*v3, u_field, v_field)
        self.x += (dt / 6.0) * (u1 + 2*u2 + 2*u3 + u4)
        self.y += (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        self._enforce_boundaries()
        self.history.append(np.column_stack((self.x, self.y)))

    def save_trajectories(self, filename: str):
        """Exporta el historial completo de trayectorias."""
        np.savez(filename, trajectories=np.array(self.history))
