import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    snapshots_dir = output_dir / "snapshots"
    campos_dir = snapshots_dir / "campos"
    psi_dir = snapshots_dir / "psi"
    velocidad_dir = snapshots_dir / "velocidad"
    completas_dir = snapshots_dir / "completas"
    modelos_dir = output_dir / "modelos"
    videos_dir = output_dir / "videos"
    metricas_dir = output_dir / "metricas"
    
    # Crear todas las carpetas
    for d in [output_dir, snapshots_dir, campos_dir, psi_dir, velocidad_dir, 
              completas_dir, modelos_dir, videos_dir, metricas_dir]:
        os.makedirs(d, exist_ok=True)
    
    plt_qg.set_plot_style(profile='presentation')
    
    # 2. Inicialización
    model = QGModel(p, n)
    tracker = ParticleTracker(model.grid, n_particles=500)
    
    z0 = create_dipole_ic(model.grid, p)
    model.set_initial_condition(z0)
    tracker.seed_cloud(p.lx/2, p.ly/2, radius=4e4, seed=42)
    
    # 3. Bucle de Simulación
    total_days = 40
    steps_per_day = int(86400 / n.dt)
    total_steps = total_days * steps_per_day
    save_interval = steps_per_day
    animation_interval = max(1, steps_per_day // 6)
    
    print(f"--- Simulación TFG: {total_days} días ---")

    field_zeta_frames = []
    field_psi_frames = []
    field_days = []
    anim_zeta_frames = []
    anim_psi_frames = []
    anim_u_frames = []
    anim_v_frames = []
    anim_px_frames = []
    anim_py_frames = []
    anim_days = []
    energy_series = []
    
    # Guardar estado inicial (t=0)
    t_days = 0.0
    print(f"Día {t_days:.1f}: Guardando snapshot inicial...")
    model.save_state(str(modelos_dir / "model_0000.npz"))
    fig, _ = plt_qg.plot_snapshot_complete(model.grid, model.zeta, model.psi, 
                                          tracker.x, tracker.y, t_days,
                                          save_path=str(completas_dir / "snap_0000.png"))
    plt.close(fig)
    fig, _ = plt_qg.plot_fields(model.grid, model.zeta, model.psi,
                                title="Campos Eulerianos - Día 0.00",
                                save_path=str(campos_dir / "fields_0000.png"))
    plt.close(fig)
    fig, _ = plt_qg.plot_streamfunction(
        model.grid,
        model.psi,
        title="Función de Corriente - Día 0.00",
        save_path=str(psi_dir / "psi_0000.png"),
    )
    plt.close(fig)
    fig, _ = plt_qg.plot_velocity_field(
        model.grid,
        model.u,
        model.v,
        save_path=str(velocidad_dir / "vel_0000.png"),
    )
    plt.close(fig)

    field_zeta_frames.append(model.zeta.copy())
    field_psi_frames.append(model.psi.copy())
    field_days.append(t_days)
    anim_zeta_frames.append(model.zeta.copy())
    anim_psi_frames.append(model.psi.copy())
    anim_u_frames.append(model.u.copy())
    anim_v_frames.append(model.v.copy())
    anim_px_frames.append(tracker.x.copy())
    anim_py_frames.append(tracker.y.copy())
    anim_days.append(t_days)
    energy_series.append(diag.kinetic_energy(model.u, model.v, model.grid.dx, model.grid.dy))

    # Avance temporal
    for s in tqdm(range(1, total_steps + 1), total=total_steps, desc="Simulación QG", unit="step", dynamic_ncols=True):
        model.step()
        tracker.step(n.dt, model.u, model.v)

        if s % animation_interval == 0 or s == total_steps:
            anim_zeta_frames.append(model.zeta.copy())
            anim_psi_frames.append(model.psi.copy())
            anim_u_frames.append(model.u.copy())
            anim_v_frames.append(model.v.copy())
            anim_px_frames.append(tracker.x.copy())
            anim_py_frames.append(tracker.y.copy())
            anim_days.append(model.t / 86400.0)
        
        energy_series.append(diag.kinetic_energy(model.u, model.v, model.grid.dx, model.grid.dy))

        if s % save_interval == 0:
            t_days = model.t / 86400.0
            
            model.save_state(str(modelos_dir / f"model_{s:04d}.npz"))
            fig, _ = plt_qg.plot_snapshot_complete(
                model.grid, model.zeta, model.psi, 
                tracker.x, tracker.y, t_days,
                save_path=str(completas_dir / f"snap_{s:04d}.png")
            )
            plt.close(fig)
            fig, _ = plt_qg.plot_fields(
                model.grid, model.zeta, model.psi,
                title=f"Campos Eulerianos - Día {t_days:.2f}",
                save_path=str(campos_dir / f"fields_{s:04d}.png")
            )
            plt.close(fig)
            fig, _ = plt_qg.plot_streamfunction(
                model.grid,
                model.psi,
                title=f"Función de Corriente - Día {t_days:.2f}",
                save_path=str(psi_dir / f"psi_{s:04d}.png"),
            )
            plt.close(fig)
            fig, _ = plt_qg.plot_velocity_field(
                model.grid,
                model.u,
                model.v,
                save_path=str(velocidad_dir / f"vel_{s:04d}.png"),
            )
            plt.close(fig)
            field_zeta_frames.append(model.zeta.copy())
            field_psi_frames.append(model.psi.copy())
            field_days.append(t_days)

    # 4. Análisis y Métricas
    print("\nSimulación finalizada. Generando resultados finales...")
    
    trajectories = np.array(tracker.history)
    tracker.save_trajectories(str(modelos_dir / "trajectories.npz"))
    
    msd_series = diag.msd(trajectories)
    r_rms_series = diag.rms_radius_timeseries(trajectories)
    
    # Eje temporal exacto en días
    t_axis = np.arange(len(msd_series)) * n.dt / 86400.0
    
    fig_metrics = plt_qg.plot_dispersion_metrics(t_axis, msd_series, r_rms_series, 
                                                save_path=str(metricas_dir / "metrics.svg"))
    plt.close(fig_metrics)
    fig_energy = plt_qg.plot_energy_timeseries(
        t_axis,
        np.array(energy_series),
        save_path=str(metricas_dir / "energy.svg"),
    )
    plt.close(fig_energy)

    # 5. Animaciones MP4
    print("Generando animaciones MP4...")
    plt_qg.save_complete_animation(
        model.grid,
        anim_zeta_frames,
        anim_psi_frames,
        anim_px_frames,
        anim_py_frames,
        anim_days,
        save_path=str(videos_dir / "evolution_complete.mp4"),
        fps=8,
    )
    plt_qg.save_fields_animation(
        model.grid,
        field_zeta_frames,
        field_psi_frames,
        field_days,
        save_path=str(videos_dir / "fields_evolution.mp4"),
        fps=6,
    )
    plt_qg.save_streamfunction_animation(
        model.grid,
        anim_psi_frames,
        anim_days,
        save_path=str(videos_dir / "psi_evolution.mp4"),
        fps=4,
    )
    plt_qg.save_velocity_animation(
        model.grid,
        anim_u_frames,
        anim_v_frames,
        anim_days,
        save_path=str(videos_dir / "velocity_evolution.mp4"),
        fps=6,
    )
    
    report = diag.summary_report(trajectories)
    print("="*50)
    print(f"RESULTADOS DEL EXPERIMENTO:")
    print(f"MSD Final: {report['msd_final']:.2e} m^2")
    print(f"Crecimiento nube (R_rms): {report['expansion_factor']:.2f}")
    print(f"Desplazamiento neto: {report['net_displacement']/1e3:.2f} km")
    print(f"\nESTRUCTURA DE OUTPUTS:")
    print(f"  📁 output/")
    print(f"     ├── snapshots/")
    print(f"     │   ├── campos/      (Vorticidad)")
    print(f"     │   ├── psi/         (Función de Corriente)")
    print(f"     │   ├── velocidad/   (Campos de Velocidad)")
    print(f"     │   └── completas/   (Vorticidad + Partículas)")
    print(f"     ├── modelos/         (Estados .npz + trayectorias)")
    print(f"     ├── videos/          (MP4: complete, fields, psi, velocity)")
    print(f"     └── metricas/        (Gráficas: metrics, energy)")
    print(f"\nTodos los formatos vectoriales (SVG) listos para presentación")
    print("="*50)

if __name__ == "__main__":
    main()
