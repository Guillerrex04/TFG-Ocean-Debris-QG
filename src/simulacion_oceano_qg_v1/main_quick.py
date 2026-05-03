import os
import numpy as np
import matplotlib.pyplot as plt

# Imports absolutos locales
from config import PhysicsConfig, NumericalConfig
from grid import Grid
from qg_model import QGModel
from particles import ParticleTracker
import diagnostics as diag
import plotting as plt_qg

def create_dipole_ic(grid, p):
    """Genera un dipolo de vorticidad (anomalía positiva y negativa)."""
    x0, y0 = p.lx * 0.4, p.ly * 0.5
    x1, y1 = p.lx * 0.6, p.ly * 0.5
    sigma = 6e4
    amp = 3e-5
    
    z_pos =  amp * np.exp(-((grid.X - x0)**2 + (grid.Y - y0)**2) / (2 * sigma**2))
    z_neg = -amp * np.exp(-((grid.X - x1)**2 + (grid.Y - y1)**2) / (2 * sigma**2))
    return z_pos + z_neg

def main():
    # 1. Configuración
    p = PhysicsConfig(nu=200.0, beta=2.0e-11)
    n = NumericalConfig(nx=129, ny=129, dt=3600.0) 
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plt_qg.set_plot_style()
    
    # 2. Inicialización
    model = QGModel(p, n)
    tracker = ParticleTracker(model.grid, n_particles=500)
    
    z0 = create_dipole_ic(model.grid, p)
    model.set_initial_condition(z0)
    tracker.seed_cloud(p.lx/2, p.ly/2, radius=4e4, seed=42)
    
    # 3. Bucle de Simulación (VERSIÓN RÁPIDA: 2 días)
    total_days = 2  # ← Reducido de 10 a 2 para prueba rápida
    steps_per_day = int(86400 / n.dt)
    total_steps = total_days * steps_per_day
    save_interval = steps_per_day
    
    print(f"--- Simulación TFG RÁPIDA: {total_days} días ---")
    
    # Guardar estado inicial (t=0)
    t_days = 0.0
    print(f"Día {t_days:.1f}: Guardando snapshot inicial...")
    model.save_state(f"{output_dir}/model_0000.npz")
    fig, _ = plt_qg.plot_snapshot_complete(model.grid, model.zeta, model.psi, 
                                          tracker.x, tracker.y, t_days,
                                          save_path=f"{output_dir}/snap_0000.png")
    plt.close(fig)

    # Avance temporal
    for s in range(1, total_steps + 1):
        model.step()
        tracker.step(n.dt, model.u, model.v)
        
        if s % save_interval == 0:
            t_days = model.t / 86400.0
            print(f"Día {t_days:.1f}: Procesando...")
            
            model.save_state(f"{output_dir}/model_{s:04d}.npz")
            fig, _ = plt_qg.plot_snapshot_complete(
                model.grid, model.zeta, model.psi, 
                tracker.x, tracker.y, t_days,
                save_path=f"{output_dir}/snap_{s:04d}.png"
            )
            plt.close(fig)

    # 4. Análisis y Métricas
    print("\nSimulación finalizada. Generando resultados finales...")
    
    trajectories = np.array(tracker.history)
    tracker.save_trajectories(f"{output_dir}/trajectories.npz")
    
    msd_series = diag.msd(trajectories)
    r_rms_series = diag.rms_radius_timeseries(trajectories)
    
    # Eje temporal exacto en días
    t_axis = np.arange(len(msd_series)) * n.dt / 86400.0
    
    fig_metrics = plt_qg.plot_dispersion_metrics(t_axis, msd_series, r_rms_series, 
                                                save_path=f"{output_dir}/metrics.png")
    plt.close(fig_metrics)
    
    report = diag.summary_report(trajectories)
    print("="*40)
    print(f"RESULTADOS DEL EXPERIMENTO:")
    print(f"MSD Final: {report['msd_final']:.2e} m^2")
    print(f"Crecimiento nube (R_rms): {report['expansion_factor']:.2f}")
    print(f"Desplazamiento neto: {report['net_displacement']/1e3:.2f} km")
    print(f"Archivos en: {output_dir}/")
    print("="*40)

if __name__ == "__main__":
    main()
