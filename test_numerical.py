#!/usr/bin/env python3
"""
Test de validacion del nucleo numerico QG.
Ejecutar: python test_numerical.py
"""

import numpy as np
from scipy import fft


def laplacian_fd(f, dx, dy):
    """Laplaciano con diferencias finitas."""
    res = np.zeros_like(f)
    res[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    return res


def solve_poisson(zeta, dx, dy):
    """Poisson: nabla^2 psi = zeta."""
    nx, ny = zeta.shape
    nx_int, ny_int = nx - 2, ny - 2
    
    interior = zeta[1:-1, 1:-1].copy()
    zeta_hat = fft.dst(fft.dst(interior, type=1, axis=0), type=1, axis=1)
    
    k_grid = np.arange(1, nx_int+1)[:, None] * np.pi / (nx_int + 1)
    l_grid = np.arange(1, ny_int+1)[None, :] * np.pi / (ny_int + 1)
    
    k_eff = 2 * np.sin(k_grid/2) / dx
    l_eff = 2 * np.sin(l_grid/2) / dy
    lambda_k = -(k_eff**2 + l_eff**2)
    
    psi_hat = zeta_hat / lambda_k
    psi_interior = fft.idst(fft.idst(psi_hat, type=1, axis=0), type=1, axis=1)
    
    psi = np.zeros_like(zeta)
    psi[1:-1, 1:-1] = psi_interior
    return psi


def test_poisson():
    """Test del solver de Poisson."""
    nx, ny = 513, 257
    Lx, Ly = 4e6, 2e6
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    nx_int, ny_int = nx - 2, ny - 2
    
    errors = []
    for kx in [1, 2, 3, 5, 10]:
        for ky in [1, 2, 3, 5, 10]:
            k_grid = kx * np.pi / (nx_int + 1)
            l_grid = ky * np.pi / (ny_int + 1)
            
            x_idx = np.arange(1, nx-1)
            y_idx = np.arange(1, ny-1)
            
            psi_ana = np.zeros((nx, ny))
            psi_ana[1:-1, 1:-1] = np.sin(k_grid * x_idx[:, None]) * np.sin(l_grid * y_idx[None, :])
            
            zeta = laplacian_fd(psi_ana, dx, dy)
            psi_sol = solve_poisson(zeta, dx, dy)
            
            error = np.max(np.abs(psi_ana[1:-1, 1:-1] - psi_sol[1:-1, 1:-1]))
            denom = np.max(np.abs(psi_ana[1:-1, 1:-1]))
            errors.append(error / denom if denom > 1e-14 else error)
    
    max_error = np.max(errors)
    print(f"TEST Poisson: max error = {max_error:.2e}")
    return max_error < 1e-12


def test_velocities():
    """Test de velocidades: u = -dpsi/dy, v = dpsi/dx."""
    nx, ny = 513, 257
    Lx, Ly = 4e6, 2e6
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    
    k = 10 * 2*np.pi / Lx
    l = 5 * 2*np.pi / Ly
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    psi = np.sin(k*X) * np.sin(l*Y)
    
    u_num = np.zeros_like(psi)
    v_num = np.zeros_like(psi)
    u_num[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
    v_num[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    
    u_ana = -l * np.cos(k*X) * np.sin(l*Y)
    v_ana =  k * np.sin(k*X) * np.cos(l*Y)
    
    u_error = np.max(np.abs(u_num[:, 1:-1] - u_ana[:, 1:-1]))
    v_error = np.max(np.abs(v_num[1:-1, :] - v_ana[1:-1, :]))
    
    print(f"TEST Velocidades: u error = {u_error:.2e}, v error = {v_error:.2e}")
    return u_error < 1e-4 and v_error < 1e-4


if __name__ == "__main__":
    print("=" * 50)
    print("TEST NUMERICO QG")
    print("=" * 50)
    
    t1 = test_poisson()
    t2 = test_velocities()
    
    print("=" * 50)
    if t1 and t2:
        print("TODOS LOS TESTS PASARON")
    else:
        print("ALGUNOS TESTS FALLARON")
    print("=" * 50)