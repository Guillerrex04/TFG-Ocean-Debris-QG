import numpy as np

def kinetic_energy(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    """Energía cinética integrada en el dominio (J/kg en 2D por unidad de densidad)."""
    return 0.5 * np.sum(u**2 + v**2) * dx * dy

def centroid(positions: np.ndarray) -> np.ndarray:
    """Calcula el centro de masas (x_medio, y_medio) de una nube (N, 2)."""
    return np.mean(positions, axis=0)

def centroid_timeseries(trajectories: np.ndarray) -> np.ndarray:
    """Calcula la evolución del centroide para trayectorias (T, N, 2)."""
    return np.mean(trajectories, axis=1)

def msd(trajectories: np.ndarray) -> np.ndarray:
    """Calcula el MSD (T, N, 2) -> (T,)."""
    pos0 = trajectories[0]
    squared_dist = np.sum((trajectories - pos0)**2, axis=-1)
    return np.mean(squared_dist, axis=1)

def rms_radius_timeseries(trajectories: np.ndarray) -> np.ndarray:
    """Calcula la evolución del radio RMS respecto al centroide (T, N, 2)."""
    T, N, _ = trajectories.shape
    r_rms = np.zeros(T)
    for t in range(T):
        r_rms[t] = dispersion_stats(trajectories[t])['r_rms']
    return r_rms

def dispersion_stats(positions: np.ndarray) -> dict:
    """Calcula estadísticas detalladas de dispersión para una instantánea (N, 2)."""
    x, y = positions[:, 0], positions[:, 1]
    cx, cy = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    return {
        "centroid": (cx, cy),
        "var_x": var_x,
        "var_y": var_y,
        "std_x": np.sqrt(var_x),
        "std_y": np.sqrt(var_y),
        "r_rms": np.sqrt(np.mean((x - cx)**2 + (y - cy)**2)),
        "bbox": (np.min(x), np.max(x), np.min(y), np.max(y))
    }

def summary_report(trajectories: np.ndarray) -> dict:
    """Resumen comparativo entre el estado inicial y final."""
    stats0, statsf = dispersion_stats(trajectories[0]), dispersion_stats(trajectories[-1])
    return {
        "msd_final": msd(trajectories)[-1],
        "expansion_factor": statsf['r_rms'] / stats0['r_rms'] if stats0['r_rms'] > 0 else 1.0,
        "net_displacement": np.linalg.norm(np.array(statsf['centroid']) - np.array(stats0['centroid'])),
        "final_stats": statsf
    }
