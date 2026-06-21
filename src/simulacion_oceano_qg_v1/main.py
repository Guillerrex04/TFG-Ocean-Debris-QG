import os
import glob

def _generate_codigo_completo():
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codigo_completo.txt")
    py_files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py")))
    with open(output_file, "w", encoding="utf-8") as out:
        for filepath in py_files:
            filename = os.path.basename(filepath)
            out.write(f"{'='*60}\n")
            out.write(f"ARCHIVO: {filename}\n")
            out.write(f"{'='*60}\n\n")
            with open(filepath, "r", encoding="utf-8") as f:
                out.write(f.read())
            out.write("\n\n")

_generate_codigo_completo()

from helpers import setup_nvidia_dlls
setup_nvidia_dlls()

import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected")

import numpy as np
import cupy as cp
from tqdm import tqdm
import time

from config import PhysicsConfig, NumericalConfig
from qg_model import QGModel
from helpers import get_grid_metadata
from diagnostics import enstrophy_integrated, diagnostic_summary
from diagnostics import search_area, dispersion_stats
from particles import ParticleTracker


def create_perturbed_ic(grid, amp: float = 1e-5, seed: int = 42):
    nx, ny = grid.nx, grid.ny
    rng = np.random.default_rng(seed)
    zeta0 = rng.uniform(-amp, amp, size=(nx, ny)).astype(np.float64)
    zeta0[0, :] = 0.0
    zeta0[-1, :] = 0.0
    zeta0[:, 0] = 0.0
    zeta0[:, -1] = 0.0
    return zeta0


def run_single_simulation(p, n, release_day, output_dir, suffix, total_days=2000):
    """Ejecuta una simulación QG completa y guarda resultados con sufijo."""
    t_start = time.time()

    modelos_dir = os.path.join(output_dir, f"modelos{suffix}")
    os.makedirs(modelos_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"SIMULACIÓN {suffix} — N={n.n_particles} partículas")
    print("="*60)
    print(f"  tau0={p.tau0}, nu={p.nu}, r={p.r}, H={p.H}")
    print(f"  Lx={p.lx/1e6:.1f}Mm, Ly={p.ly/1e6:.1f}Mm")
    print(f"  nx={n.nx}, ny={n.ny}")

    model = QGModel(p, n)

    wind_curl_cpu = cp.asnumpy(model.grid.wind_curl)
    wind_forcing_cpu = cp.asnumpy(model.wind_forcing)
    cx, cy = n.nx // 2, n.ny // 2
    print(f"\n  === DIAGNOSTICO DE VIENTO ===")
    print(f"  wind_curl[{cx},{cy}] = {wind_curl_cpu[cx, cy]:.4e}")
    print(f"  wind_forcing[{cx},{cy}] = {wind_forcing_cpu[cx, cy]:.4e}")
    print(f"  wind_forcing max = {float(np.max(np.abs(wind_forcing_cpu))):.4e}")
    print(f"  wind_forcing mean = {float(np.mean(np.abs(wind_forcing_cpu))):.4e}")
    print(f"  =============================")

    z0 = create_perturbed_ic(model.grid, amp=1e-5, seed=42)
    print(f"\n  === DIAGNOSTICO IC (ANTES DE set_initial_condition) ===")
    print(f"  z0 rms = {float(np.sqrt(np.mean(z0**2))):.4e}")
    print(f"  z0 max = {float(np.max(np.abs(z0))):.4e}")
    print(f"  z0 interior mean = {float(np.mean(z0[1:-1, 1:-1])):.4e}")
    print(f"  z0 interior max = {float(np.max(np.abs(z0[1:-1, 1:-1]))):.4e}")
    print(f"  =======================================================")

    model.set_initial_condition(z0)

    psi_cpu = model.psi.get()
    zeta_cpu = model.zeta.get()
    u_cpu = model.u.get()
    v_cpu = model.v.get()

    lap_psi = cp.asnumpy(model.ops.laplacian(model.psi))
    lap_psi_int = lap_psi[1:-1, 1:-1]
    zeta_int = zeta_cpu[1:-1, 1:-1]
    lap_max = float(np.max(np.abs(lap_psi_int)))
    zeta_max = float(np.max(np.abs(zeta_int)))
    ratio = lap_max / max(zeta_max, 1e-20)

    correlacion = float(np.sum(lap_psi_int * zeta_int))
    signo_ok = correlacion > 0.0

    print(f"\n  === VALIDACION POISSON ===")
    print(f"  Lap(psi) max = {lap_max:.4e}")
    print(f"  zeta max     = {zeta_max:.4e}")
    print(f"  ratio        = {ratio:.6f}")
    print(f"  signo ∇²ψ·ζ  = {'POSITIVO ✓' if signo_ok else 'NEGATIVO ✗ (INVERTIDO)'}")
    if not np.isclose(ratio, 1.0, rtol=0.1):
        print(f"\n  ERROR: ratio = {ratio:.4f} != 1.0")
        print(f"  El solver de Poisson esta MAL escalado.")
        raise RuntimeError(f"Poisson solver FAILED: ratio={ratio:.4f}")
    if not signo_ok:
        print(f"\n  ERROR DE SIGNO: ∇²ψ y ζ tienen signos opuestos.")
        print(f"  La relacion zeta = nabla^2 psi NO se cumple con el signo correcto.")
        print(f"  Revisa la convencion espectral en PoissonSolver.")
        raise RuntimeError("Poisson solver SIGN ERROR: ∇²ψ = -ζ en vez de ∇²ψ = +ζ")
    print(f"  Poisson OK (ratio ~= 1.0, signo correcto)")
    print(f"  ==========================")

    ke_init = model.get_kinetic_energy()
    print(f"\n  === DIAGNOSTICO IC (TRAS set_initial_condition) ===")
    print(f"  model.zeta rms = {float(np.sqrt(np.mean(zeta_cpu**2))):.4e}")
    print(f"  model.zeta max = {float(np.max(np.abs(zeta_cpu))):.4e}")
    print(f"  model.psi max  = {float(np.max(np.abs(psi_cpu))):.4e}")
    print(f"  model.u max    = {float(np.max(np.abs(u_cpu))):.4e}")
    print(f"  model.v max    = {float(np.max(np.abs(v_cpu))):.4e}")
    print(f"  KE inicial     = {ke_init:.4e}")
    print(f"  ====================================================")

    cfl_dt = model.get_cfl_dt()
    print(f"\n  dt inicial: {n.dt:.0f}s | CFL dt: {cfl_dt:.0f}s")
    model.adjust_dt()
    print(f"  dt efectivo: {model.dt:.0f}s")

    particles_initialized = False
    tracker = None

    total_time = total_days * 86400.0
    save_interval_days = n.save_interval_days

    print(f"\n--- Simulacion: {total_days} dias ({total_time:.0f}s) ---")
    print(f"  dt: {model.dt:.0f}s | save_interval: {save_interval_days} dias")
    t_physics = time.time()

    energy_times = []
    energy_values = []
    enstrophy_times = []
    enstrophy_values = []
    snapshot_days = []
    snapshot_steps = []
    search_areas = []
    centroids = []
    rms_radii = []
    metrics_days = []

    model.save_state(os.path.join(modelos_dir, "model_0000.npz"))
    energy_times.append(0.0)
    energy_values.append(ke_init)
    enstrophy_times.append(0.0)
    enstrophy_values.append(enstrophy_integrated(zeta_cpu, model.grid.dx, model.grid.dy))
    snapshot_days.append(0.0)
    snapshot_steps.append(0)

    calc_time = 0.0
    prev_dt = model.dt
    step = 0

    next_snapshot_time = save_interval_days * 86400.0
    energy_interval_days = 0.1
    next_energy_time = energy_interval_days * 86400.0
    next_print_time = save_interval_days * 86400.0

    with tqdm(total=total_days, desc=f"QG{suffix}", unit="day", dynamic_ncols=True) as pbar:
        while model.t < total_time:
            step += 1
            t0 = time.time()
            model.step()
            model.adjust_dt()
            calc_time += time.time() - t0

            t_days = model.t / 86400.0
            dias_redondeados = round(t_days, 2)
            incremento = dias_redondeados - pbar.n
            pbar.set_description(f'{suffix} Dia {dias_redondeados:.2f}')
            pbar.update(incremento)

            if model.t >= release_day * 86400.0 and not particles_initialized:
                x0 = model.grid.lx / 2.0
                y0 = model.grid.ly / 2.0
                tracker = ParticleTracker(model.grid, n.n_particles)
                tracker.seed_cloud(x0, y0, radius=n.release_radius, seed=42)
                particles_initialized = True
                tqdm.write(f"[ACCIDENTE{suffix}] Día {release_day}: {n.n_particles} partículas liberadas en ({x0/1e3:.0f}, {y0/1e3:.0f}) km")

            if particles_initialized:
                u_cpu = model.u.get() if hasattr(model.u, 'get') else model.u
                v_cpu = model.v.get() if hasattr(model.v, 'get') else model.v
                tracker.step(model.dt, u_cpu, v_cpu)

            if step % 10 == 0:
                current_dt = model.dt
                if current_dt < prev_dt * 0.5:
                    tqdm.write(f"[ALERTA CFL{suffix}] Paso {step}: dt {prev_dt:.0f}s -> {current_dt:.0f}s")
                prev_dt = current_dt

            if model.t >= next_energy_time:
                energy_times.append(t_days)
                energy_values.append(model.get_kinetic_energy())
                enstrophy_times.append(t_days)
                zeta_cpu = model.zeta.get() if hasattr(model.zeta, 'get') else model.zeta
                enstrophy_values.append(enstrophy_integrated(zeta_cpu, model.grid.dx, model.grid.dy))
                next_energy_time += energy_interval_days * 86400.0

            if model.t >= next_print_time:
                ke = model.get_kinetic_energy()
                zeta_cpu = model.zeta.get() if hasattr(model.zeta, 'get') else model.zeta
                u_max = float(np.max(np.abs(model.u.get())))
                v_max = float(np.max(np.abs(model.v.get())))
                zeta_max = float(np.max(np.abs(zeta_cpu)))
                tqdm.write(f"  [{suffix} Dia {t_days:.1f}] KE={ke:.4e} | u_max={u_max:.4e} | v_max={v_max:.4e} | zeta_max={zeta_max:.4e}")
                next_print_time += save_interval_days * 86400.0

            while model.t >= next_snapshot_time:
                particles_kw = {}
                if particles_initialized:
                    particles_kw = {'particles_x': tracker.x, 'particles_y': tracker.y}
                    area_m2 = search_area(np.column_stack((tracker.x, tracker.y)))
                    stats = dispersion_stats(np.column_stack((tracker.x, tracker.y)))
                    search_areas.append(area_m2 / 1e6)
                    centroids.append(stats['centroid'])
                    rms_radii.append(stats['r_rms'])
                    metrics_days.append(t_days)
                    tqdm.write(f"  [{suffix} Día {t_days:.1f}] Área={area_m2/1e6:.2f} km² | RMS={stats['r_rms']:.2f} m")
                model.save_state(os.path.join(modelos_dir, f"model_{step:05d}.npz"), **particles_kw)
                snapshot_days.append(t_days)
                snapshot_steps.append(step)
                next_snapshot_time += save_interval_days * 86400.0

    t_final = model.t / 86400.0

    last_snapshot_day = snapshot_days[-1] if snapshot_days else -1.0
    if t_final - last_snapshot_day > 1e-6:
        particles_kw = {}
        if particles_initialized:
            particles_kw = {'particles_x': tracker.x, 'particles_y': tracker.y}
            area_m2 = search_area(np.column_stack((tracker.x, tracker.y)))
            stats = dispersion_stats(np.column_stack((tracker.x, tracker.y)))
            search_areas.append(area_m2 / 1e6)
            centroids.append(stats['centroid'])
            rms_radii.append(stats['r_rms'])
            metrics_days.append(t_final)
        model.save_state(os.path.join(modelos_dir, f"model_{step:05d}.npz"), **particles_kw)
        snapshot_days.append(t_final)
        snapshot_steps.append(step)

    t_physics_end = time.time()

    summary_data = {
        'snapshot_days': np.array(snapshot_days),
        'snapshot_steps': np.array(snapshot_steps),
        'energy_days': np.array(energy_times),
        'energy': np.array(energy_values),
        'enstrophy_days': np.array(enstrophy_times),
        'enstrophy': np.array(enstrophy_values),
        'release_day': float(release_day),
        'lx': float(model.grid.lx),
        'ly': float(model.grid.ly),
        'nx': int(model.grid.nx),
        'ny': int(model.grid.ny),
        'dx': float(model.grid.dx),
        'dy': float(model.grid.dy),
        'center_x': float(model.grid.lx / 2),
        'center_y': float(model.grid.ly / 2),
        'origin': 'southwest',
        'x_positive': 'eastward',
        'y_positive': 'northward',
        'coordinates': 'physical_with_centered_available',
        'spatial_convention': 'origin_southwest_x_eastward_y_northward',
        'n_particles': n.n_particles,
    }
    if search_areas:
        summary_data['metrics_days'] = np.array(metrics_days)
        summary_data['area_km2'] = np.array(search_areas)
        summary_data['centroids'] = np.array(centroids)
        summary_data['rms_dispersion'] = np.array(rms_radii)

    summary_path = os.path.join(output_dir, f"simulation_summary{suffix}.npz")
    np.savez(summary_path, **summary_data)

    print("\n" + "="*60)
    print(f"RESULTADOS {suffix} ({total_days} dias):")
    print(f"  KE inicial: {energy_values[0]:.4e}")
    print(f"  KE dia 1:   {next((e for t, e in zip(energy_times, energy_values) if t >= 1.0), energy_values[0]):.4e}")
    print(f"  KE final:  {energy_values[-1]:.4e}")
    print(f"  Enstrofia final: {enstrophy_values[-1]:.4e}")
    print(f"  Dias simulados: {t_final:.2f}")
    print(f"  Pasos ejecutados: {step}")
    print(f"  tau0={p.tau0}, nu={p.nu}, r={p.r}, H={p.H}")
    print(f"  amp_IC=1e-5, save_interval={save_interval_days}dias -> {len(snapshot_days)} snapshots")
    print(f"  wind_forcing max = {float(np.max(np.abs(wind_forcing_cpu))):.4e} s^-2")
    print("="*60)

    informe_path = os.path.join(output_dir, f"informe_dispersion{suffix}.txt")
    with open(informe_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("INFORME DE DISPERSIÓN - ACCIDENTE AÉREO\n")
        f.write("=" * 60 + "\n\n")
        if particles_initialized and search_areas:
            f.write(f"Día de liberación de restos:      {release_day:.1f}\n")
            f.write(f"Número de partículas:             {n.n_particles}\n")
            f.write(f"Días totales simulados:           {total_days:.1f}\n")
            f.write(f"Días simulados post-accidente:    {t_final - release_day:.1f}\n")
            f.write(f"Área final de búsqueda:           {search_areas[-1]:.2f} km²\n")
            f.write(f"Radio RMS final:                  {rms_radii[-1]:.2f} m\n")
            cx_last, cy_last = centroids[-1]
            f.write(f"Centroide final (X, Y):           ({cx_last:.1f}, {cy_last:.1f}) m\n")
        else:
            f.write("Las partículas nunca se activaron durante la simulación.\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"\n  Informe guardado: {informe_path}")

    t_end = time.time()
    physics_time_calc = t_physics_end - t_physics
    print(f"  Fisica: {physics_time_calc:.1f}s ({physics_time_calc/60:.1f}min)")
    print(f"  Total:  {t_end - t_start:.1f}s ({(t_end - t_start)/60:.1f}min)")
    print(f"  Ratio:  {t_final/physics_time_calc:.1f} dias/s")


def main():
    t_start = time.time()

    p = PhysicsConfig()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("QG MODEL V4.6 - TRES SIMULACIONES CONSECUTIVAS")
    print("="*60)
    print(f"  {10}, {100} y {1000} partículas con distribución Gaussiana")
    print(f"  sigma = release_radius / 2  (95% dentro de {NumericalConfig().release_radius:.0f} m)")
    print("="*60)

    release_day = float(input("Introduce el día de liberación del accidente (ej. 500): "))

    configs = [
        (10,  "_res_10"),
        (100, "_res_100"),
        (1000,"_res_1000"),
    ]

    for n_particles, suffix in configs:
        n = NumericalConfig(n_particles=n_particles)
        run_single_simulation(p, n, release_day, output_dir, suffix)

    t_end = time.time()
    print("\n" + "="*60)
    print("TRES SIMULACIONES COMPLETADAS")
    print(f"  Tiempo total: {(t_end - t_start)/60:.1f} min")
    print("="*60)


if __name__ == "__main__":
    main()