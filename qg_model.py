import cupy as cp
import numpy as np
from grid import Grid
from operators import Operators
from poisson import PoissonSolver
from config import PhysicsConfig, NumericalConfig

DTYPE = cp.float64


class QGModel:
    """QG barotrópico con forzamiento de viento - Optimizado HPC."""
    
    def __init__(self, p_cfg: PhysicsConfig, n_cfg: NumericalConfig):
        self.p = p_cfg
        self.n = n_cfg
        self.cfl_safety = n_cfg.cfl_safety
        self.dt_target = n_cfg.dt
        
        self.grid = Grid(p_cfg, n_cfg)
        self.ops = Operators(self.grid)
        self.poisson = PoissonSolver(self.grid)
        
        self.zeta = cp.zeros(self.grid.shape, dtype=DTYPE)
        self.psi = cp.zeros(self.grid.shape, dtype=DTYPE)
        self.u = cp.zeros(self.grid.shape, dtype=DTYPE)
        self.v = cp.zeros(self.grid.shape, dtype=DTYPE)
        
        self.beta_y = self.grid.get_beta_y(self.p.beta).astype(DTYPE)
        self.grid.wind_curl = self.grid.get_wind_stress_curl(self.p).astype(DTYPE)
        self.wind_forcing = (self.grid.wind_curl / (self.p.rho0 * self.p.H)).astype(DTYPE)
        self.t = cp.float64(0.0)
        
        self._dt = cp.float64(self.dt_target)
    
    def _update_diagnostics(self):
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.u[:, 1:-1] = -(self.psi[:, 2:] - self.psi[:, :-2]) / (cp.float64(2.0) * self.grid.dy)
        self.v[1:-1, :] = (self.psi[2:, :] - self.psi[:-2, :]) / (cp.float64(2.0) * self.grid.dx)
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, -1] = 0.0
    
    def get_max_speed(self) -> float:
        speed = cp.sqrt(self.u**2 + self.v**2)
        return float(cp.max(speed).get())
    
    def get_cfl_dt_advective(self, safety: float = None) -> float:
        if safety is None:
            safety = self.cfl_safety
        umax = max(self.get_max_speed(), 1e-10)
        dx_min = min(self.grid.dx, self.grid.dy)
        return safety * dx_min / umax
    
    def get_cfl_dt_diffusive(self, safety: float = None) -> float:
        if safety is None:
            safety = self.cfl_safety
        nu = float(self.p.nu)
        if nu <= 0:
            return float('inf')
        dx2 = self.grid.dx ** 2
        dy2 = self.grid.dy ** 2
        return safety * 0.25 * min(dx2, dy2) / nu
    
    def get_cfl_dt(self, safety: float = None) -> float:
        dt_adv = self.get_cfl_dt_advective(safety)
        dt_diff = self.get_cfl_dt_diffusive(safety)
        return min(dt_adv, dt_diff)
    
    def get_kinetic_energy(self) -> float:
        speed_sq = self.u**2 + self.v**2
        energy = cp.sum(speed_sq) * self.grid.dx * self.grid.dy * cp.float64(0.5)
        return float(energy.get())
    
    def get_tendency(self, zeta: cp.ndarray, psi_guess: cp.ndarray = None):
        psi_stage, _ = self.poisson.solve(zeta, psi_guess=psi_guess)
        q = zeta + self.beta_y
        adv = -self.ops.arakawa_jacobian(psi_stage, q)
        diff = cp.float64(self.p.nu) * self.ops.laplacian(zeta)
        wind = self.wind_forcing
        bottom_drag = -cp.float64(self.p.r) * zeta
        return adv + diff + wind + bottom_drag
    
    def step(self):
        dt = self._dt
        z0 = self.zeta
        half_dt = dt * cp.float64(0.5)
        sixth_dt = dt / cp.float64(6.0)
        
        k1 = self.get_tendency(z0, psi_guess=self.psi)
        k2 = self.get_tendency(z0 + half_dt * k1, psi_guess=self.psi)
        k3 = self.get_tendency(z0 + half_dt * k2, psi_guess=self.psi)
        k4 = self.get_tendency(z0 + dt * k3, psi_guess=self.psi)
        
        self.zeta = z0 + sixth_dt * (k1 + cp.float64(2.0)*k2 + cp.float64(2.0)*k3 + k4)
        self.t += dt
        
        self.psi, _ = self.poisson.solve(self.zeta, psi_guess=self.psi)
        self._update_diagnostics()
    
    def adjust_dt(self):
        cfl_dt = self.get_cfl_dt()
        self._dt = cp.float64(min(cfl_dt, self.dt_target))
    
    def set_initial_condition(self, zeta0):
        zeta0_np = np.asarray(zeta0, dtype=np.float64) if isinstance(zeta0, np.ndarray) else np.array(zeta0, dtype=np.float64)
        zeta0_gpu = cp.asarray(zeta0_np, dtype=DTYPE)
        self.zeta = zeta0_gpu.copy()
        self.psi, _ = self.poisson.solve(self.zeta)
        self._update_diagnostics()
        self.t = cp.float64(0.0)
        self._dt = cp.float64(self.dt_target)
    
    def save_state(self, filename: str, fields_only: bool = True, **extra_fields):
        zeta_cpu = self.zeta.get() if isinstance(self.zeta, cp.ndarray) else self.zeta
        psi_cpu = self.psi.get() if isinstance(self.psi, cp.ndarray) else self.psi
        
        cx, cy = self.grid.get_domain_center()
        metadata = {
            't': float(self.t),
            't_days': float(self.t / 86400.0),
            'zeta': zeta_cpu,
            'psi': psi_cpu,
            'lx': float(self.grid.lx),
            'ly': float(self.grid.ly),
            'nx': int(self.grid.nx),
            'ny': int(self.grid.ny),
            'dx': float(self.grid.dx),
            'dy': float(self.grid.dy),
            'center_x': float(cx),
            'center_y': float(cy),
            'origin': 'southwest',
            'x_positive': 'eastward',
            'y_positive': 'northward',
            'coordinates': 'physical_with_centered_available',
            'spatial_convention': 'origin_southwest_x_eastward_y_northward',
        }
        
        if not fields_only:
            u_cpu = self.u.get() if isinstance(self.u, cp.ndarray) else self.u
            v_cpu = self.v.get() if isinstance(self.v, cp.ndarray) else self.v
            metadata['u'] = u_cpu
            metadata['v'] = v_cpu
        
        metadata.update(extra_fields)
        np.savez(filename, **metadata)
    
    @property
    def dt(self) -> float:
        return float(self._dt)