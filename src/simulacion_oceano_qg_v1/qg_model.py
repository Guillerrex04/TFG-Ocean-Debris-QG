import numpy as np
from grid import Grid
from operators import Operators
from poisson import PoissonSolver
from config import PhysicsConfig, NumericalConfig

class QGModel:
    """Orquestador de la dinámica cuasigeostrófica barotrópica."""
    
    def __init__(self, p_cfg: PhysicsConfig, n_cfg: NumericalConfig):
        self.p = p_cfg
        self.n = n_cfg
        self.grid = Grid(p_cfg, n_cfg)
        self.ops = Operators(self.grid)
        self.poisson = PoissonSolver(self.grid)
        
        self.zeta = np.zeros(self.grid.shape)
        self.psi = np.zeros(self.grid.shape)
        self.u = np.zeros(self.grid.shape)
        self.v = np.zeros(self.grid.shape)
        
        self.beta_y = self.grid.get_beta_y(self.p.beta)
        self.t = 0.0

    def _update_diagnostics(self):
        """Calcula u, v a partir de la función de corriente actual."""
        self.u[:, 1:-1] = -(self.psi[:, 2:] - self.psi[:, :-2]) / (2 * self.grid.dy)
        self.v[1:-1, :] = (self.psi[2:, :] - self.psi[:-2, :]) / (2 * self.grid.dx)

    def get_tendency(self, zeta: np.ndarray, psi_guess: np.ndarray = None) -> np.ndarray:
        """Calcula d(zeta)/dt para un estado dado."""
        psi_stage, _ = self.poisson.solve(zeta, psi_guess=psi_guess)
        q = zeta + self.beta_y
        adv = -self.ops.arakawa_jacobian(psi_stage, q)
        diff = self.p.nu * self.ops.laplacian(zeta)
        return adv + diff

    def step(self):
        """Avanza un paso de tiempo dt usando RK4."""
        dt = self.n.dt
        z0 = self.zeta
        
        k1 = self.get_tendency(z0, psi_guess=self.psi)
        k2 = self.get_tendency(z0 + 0.5 * dt * k1, psi_guess=self.psi)
        k3 = self.get_tendency(z0 + 0.5 * dt * k2, psi_guess=self.psi)
        k4 = self.get_tendency(z0 + dt * k3, psi_guess=self.psi)
        
        self.zeta = z0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.t += dt
        
        self.psi, _ = self.poisson.solve(self.zeta, psi_guess=self.psi)
        self._update_diagnostics()

    def set_initial_condition(self, zeta0: np.ndarray):
        """Establece el campo inicial y sincroniza todas las variables."""
        self.zeta = zeta0.copy()
        self.psi, _ = self.poisson.solve(self.zeta)
        self._update_diagnostics()
        self.t = 0.0

    def save_state(self, filename: str):
        """Guarda el estado actual en un archivo .npz."""
        np.savez(filename, t=self.t, zeta=self.zeta, psi=self.psi, u=self.u, v=self.v)
