"""Módulo de utilidades comunes para el proyecto QG."""
import os
import sys


def setup_nvidia_dlls():
    """Configura el path de DLLs de NVIDIA para Windows."""
    if sys.platform != 'win32':
        return
    venv_site = os.environ.get('VIRTUAL_ENV')
    if not venv_site:
        venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'venv')
    venv_site = os.path.join(venv_site, 'Lib', 'site-packages')
    if os.path.exists(venv_site):
        for root, dirs, files in os.walk(venv_site):
            if any(f.endswith(".dll") for f in files):
                if "nvidia" in root or "cuda" in root:
                    try:
                        os.add_dll_directory(root)
                    except Exception:
                        pass


def get_grid_metadata(grid, p):
    """Genera metadatos del dominio para guardar en snapshots."""
    cx, cy = grid.lx / 2, grid.ly / 2
    return {
        'lx': float(grid.lx),
        'ly': float(grid.ly),
        'nx': int(grid.nx),
        'ny': int(grid.ny),
        'dx': float(grid.dx),
        'dy': float(grid.dy),
        'center_x': float(cx),
        'center_y': float(cy),
        'origin': 'southwest',
        'x_positive': 'eastward',
        'y_positive': 'northward',
        'coordinates': 'physical_with_centered_available',
        'spatial_convention': 'origin_southwest_x_eastward_y_northward',
        'tau0': float(p.tau0),
        'nu': float(p.nu),
        'r_friction': float(p.r),
    }