# -*- coding: utf-8 -*-
import os
import re
import gc
import sys
import warnings
warnings.filterwarnings("ignore")

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from config import PhysicsConfig
from plotting import plot_wind_stress_profile, plot_wind_curl_profile

# Always use large fonts for métricas (2K quality)
mpl.rcParams['font.size'] = 28
mpl.rcParams['axes.titlesize'] = 34
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 22
mpl.rcParams['figure.titlesize'] = 40
mpl.rcParams['lines.linewidth'] = 4.0
mpl.rcParams['axes.linewidth'] = 1.5

print("=" * 60)
print("POST-PROCESADO TFG - Selecciona qué generar")
print("=" * 60)
print("  [1] Todo (gráficas + vídeos + snapshots)")
print("  [2] Solo gráficas (SVG)")
print("  [3] Solo vídeos")
print("  [4] Solo snapshots (PNG)")
print("  [5] Gráficas + vídeos")
print("  [6] Gráficas + snapshots")
print("  [7] Vídeos + snapshots")
mode_choice = input("  Selecciona (1-7): ").strip()

RUN_PLOTS = mode_choice in ('1', '2', '5', '6')
RUN_VIDEOS = mode_choice in ('1', '3', '5', '7')
RUN_SNAPSHOTS = mode_choice in ('1', '4', '6', '7')

if RUN_VIDEOS:
    print()
    print("=" * 60)
    print("  Calidad de VÍDEO")
    print("=" * 60)
    print("  [1] Baja Calidad (720p)  -> 1280x720")
    print("  [2] Alta Calidad (2K)    -> 2560x1440")
    choice = input("  Selecciona (1/2): ").strip()
    if choice == '2':
        QUALITY = '2K'
        FIG_SIZE = (12.8, 7.2)
        DPI_SAVE = 200
    else:
        QUALITY = '720p'
        FIG_SIZE = (12.8, 7.2)
        DPI_SAVE = 100

    print()
    print("  [1] Todos los frames")
    print("  [2] Saltar cada 2 (rapidez)")
    print("  [3] Saltar cada 5 (prueba rápida)")
    skip_choice = input("  Selecciona (1/2/3): ").strip()
    if skip_choice == '2':
        FRAME_SKIP = 2
    elif skip_choice == '3':
        FRAME_SKIP = 5
    else:
        FRAME_SKIP = 1
else:
    QUALITY = '2K'
    FIG_SIZE = (12.8, 7.2)
    DPI_SAVE = 200
    FRAME_SKIP = 1

AUTHOR_TEXT = "TFG: Simulación QG Barotrópica Oceánica | Guillermo Alba Buitrón"

VMAX_ZETA_DEFAULT = 2e-5
VMAX_PSI_DEFAULT = 4e4
VMAX_VEL_DEFAULT = 0.5

GLOBAL_MAX_ZETA = VMAX_ZETA_DEFAULT
GLOBAL_MAX_PSI = VMAX_PSI_DEFAULT
GLOBAL_MAX_VEL = VMAX_VEL_DEFAULT

# Tamaños de texto para los vídeos según calidad (2K / 720p)
if QUALITY == '2K':
    VIDEO_TITLE_SIZE = 22
    AXIS_LABEL_SIZE = 18
    CBAR_LABEL_SIZE = 18
    CBAR_TICK_SIZE = 14
else:
    VIDEO_TITLE_SIZE = 18
    AXIS_LABEL_SIZE = 14
    CBAR_LABEL_SIZE = 14
    CBAR_TICK_SIZE = 11


from diagnostics import search_area as _search_area, dispersion_stats as _dispersion_stats

COMPARISON_COLORS = {10: 'blue', 100: 'green', 1000: 'red'}
COMPARISON_LABELS = {10: 'N=10', 100: 'N=100', 1000: 'N=1000'}
PARTICLE_NC_DIR = "netcdf"
PARTICLE_NC_PATTERN = "particles_output_{}.nc"


def _detect_available_counts(output_dir):
    """Devuelve lista de [10, 100, 1000] que tengan datos de partículas."""
    found = []
    for n in [10, 100, 1000]:
        # 1) Buscar archivo NetCDF
        nc_dir = os.path.join(output_dir, PARTICLE_NC_DIR)
        nc_path = os.path.join(nc_dir, PARTICLE_NC_PATTERN.format(n))
        if os.path.exists(nc_path):
            found.append(n)
            continue
        # 2) Buscar snapshots con partículas (reconstrucción desde NPZ)
        model_dir = os.path.join(output_dir, f"modelos_res_{n}")
        if os.path.isdir(model_dir):
            found.append(n)
    return found


def _load_trajectories_from_nc(n):
    """Lee trayectorias de un archivo NetCDF. Retorna (x, y) arrays shape (tiempo, partículas)."""
    from netCDF4 import Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nc_path = os.path.join(base_dir, "output", PARTICLE_NC_DIR, PARTICLE_NC_PATTERN.format(n))
    if not os.path.exists(nc_path):
        return None, None
    with Dataset(nc_path, 'r') as ds:
        x = np.asarray(ds.variables['x'][:], dtype=np.float64)
        y = np.asarray(ds.variables['y'][:], dtype=np.float64)
    return x, y


def _rebuild_trajectories_from_snapshots(n):
    """Reconstruye trayectorias desde los snapshots NPZ en output/modelos_res_{n}/."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "output", f"modelos_res_{n}")
    if not os.path.isdir(model_dir):
        return None, None
    files = sorted([f for f in os.listdir(model_dir) if f.endswith('.npz')])
    x_list, y_list = [], []
    for fname in files:
        try:
            data = np.load(os.path.join(model_dir, fname))
            px = data.get('particles_x', None)
            py = data.get('particles_y', None)
            if px is not None and py is not None and len(px) > 0:
                x_list.append(px.astype(np.float64))
                y_list.append(py.astype(np.float64))
            data.close()
        except Exception:
            pass
    if not x_list:
        return None, None
    return np.array(x_list), np.array(y_list)


def _compute_dispersion_metrics(x, y):
    """A partir de arrays (tiempo, partículas) calcula centroide, RMS, área."""
    n_frames = x.shape[0]
    t_since = np.arange(n_frames, dtype=np.float64)
    centroids = np.zeros((n_frames, 2))
    rms_disp = np.zeros(n_frames)
    area_km2 = np.zeros(n_frames)
    for t in range(n_frames):
        cx, cy = np.mean(x[t]), np.mean(y[t])
        centroids[t] = [cx, cy]
        rms_disp[t] = np.sqrt(np.mean((x[t] - cx)**2 + (y[t] - cy)**2))
        pos = np.column_stack((x[t], y[t]))
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pos)
            area_m2 = hull.volume
        except Exception:
            dx = np.max(x[t]) - np.min(x[t])
            dy = np.max(y[t]) - np.min(y[t])
            area_m2 = dx * dy
        area_km2[t] = area_m2 / 1e6
    return t_since, area_km2, rms_disp, centroids


def _load_particle_dispersion(output_dir, n):
    """Carga o reconstruye datos de dispersión para n partículas.
    
    Orden de precedencia:
      1) Archivo NetCDF en output/netcdf/particles_output_{n}.nc
      2) Reconstrucción desde snapshots NPZ en output/modelos_res_{n}/
    """
    # Intento 1: NetCDF
    nc_dir = os.path.join(output_dir, PARTICLE_NC_DIR)
    nc_path = os.path.join(nc_dir, PARTICLE_NC_PATTERN.format(n))
    if os.path.exists(nc_path):
        x, y = _load_trajectories_from_nc(n)
        if x is not None:
            t_since, area_km2, rms_disp, centroids = _compute_dispersion_metrics(x, y)
            return t_since, area_km2, rms_disp, centroids

    # Intento 2: Reconstrucción desde snapshots
    x, y = _rebuild_trajectories_from_snapshots(n)
    if x is not None:
        t_since, area_km2, rms_disp, centroids = _compute_dispersion_metrics(x, y)
        return t_since, area_km2, rms_disp, centroids

    return None, None, None, None


def _get_modelos_dir(output_dir, suffix=''):
    return os.path.join(output_dir, f"modelos{suffix}")


def _create_writer(fps):
    try:
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=20000,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium']
        )
        return writer, 'libx264 (CPU)'
    except Exception:
        writer = animation.PillowWriter(fps=fps)
        return writer, 'PillowWriter (sin FFmpeg)'

FFMPEG_AVAILABLE = False
PILLOW_AVAILABLE = False

try:
    animation.FFMpegWriter()
    FFMPEG_AVAILABLE = True
except Exception:
    print("Advertencia: ffmpeg no disponible.")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except Exception:
    print("Advertencia: Pillow no disponible.")

if not FFMPEG_AVAILABLE and not PILLOW_AVAILABLE:
    print("ERROR: No hay backend de animacion disponible.")


def _load_grid_metadata(npz_path):
    """Lee metadatos del dominio desde un snapshot .npz.
    
    Compatible hacia atrás: si el archivo antiguo no tiene los campos
    nuevos (center_x, origin, etc.), se reconstruyen con defaults.
    """
    try:
        data = np.load(npz_path)
        lx = float(data.get('lx', 6.0e6))
        ly = float(data.get('ly', 3.0e6))
        nx = int(data.get('nx', 513))
        ny = int(data.get('ny', 257))
        dx = float(data.get('dx', lx / (nx - 1)))
        dy = float(data.get('dy', ly / (ny - 1)))
        center_x = float(data.get('center_x', lx / 2))
        center_y = float(data.get('center_y', ly / 2))
        return {
            'lx': lx, 'ly': ly,
            'nx': nx, 'ny': ny,
            'dx': dx, 'dy': dy,
            'center_x': center_x,
            'center_y': center_y,
            'origin': str(data.get('origin', 'southwest')),
            'x_positive': str(data.get('x_positive', 'eastward')),
            'y_positive': str(data.get('y_positive', 'northward')),
            'coordinates': str(data.get('coordinates', 'physical_only_assumed')),
        }
    except Exception:
        return None


def _compute_grid(metadata, centered=False):
    """Genera mallas en km. Por defecto coordenadas físicas.
    
    Parameters
    ----------
    metadata : dict or None
        Diccionario con lx, ly, nx, ny y opcionalmente center_x, center_y.
    centered : bool
        Si False (default): x, y, X, Y en km físicos [0, Lx/1e3] × [0, Ly/1e3].
        Si True: devuelve coordenadas centradas (resta el centro).
    
    Returns
    -------
    x, y, X, Y : ndarrays
        Mallas en km.
    """
    if metadata is None:
        lx, ly = 6.0e6, 3.0e6
        nx, ny = 513, 257
        center_x, center_y = lx / 2, ly / 2
    else:
        lx, ly = metadata['lx'], metadata['ly']
        nx, ny = metadata['nx'], metadata['ny']
        center_x = metadata.get('center_x', lx / 2)
        center_y = metadata.get('center_y', ly / 2)
    
    x = np.linspace(0, lx / 1e3, nx)
    y = np.linspace(0, ly / 1e3, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if centered:
        cx_km = center_x / 1e3
        cy_km = center_y / 1e3
        return x - cx_km, y - cy_km, X - cx_km, Y - cy_km
    
    return x, y, X, Y


def _extract_day(filename):
    match = re.findall(r"day_(\d+\.\d+)", filename)
    if match:
        return float(match[0])
    match = re.search(r'model_(\d+)', filename)
    return int(match.group(1)) if match else 0


def _scan_global_maxima(modelos_dir, files_sorted, sample_rate=5):
    global GLOBAL_MAX_ZETA, GLOBAL_MAX_PSI, GLOBAL_MAX_VEL
    print("  Pass 1: Escaneando maximos globales (muestreo 1/{})...".format(sample_rate))
    max_zeta, max_psi, max_vel = 0.0, 0.0, 0.0
    n_scanned = 0

    for i, f in enumerate(files_sorted):
        if i % sample_rate != 0:
            continue
        try:
            data = np.load(os.path.join(modelos_dir, f))
            zeta = data['zeta']
            if np.max(np.abs(zeta)) < 1e-18:
                del data
                gc.collect()
                continue
            z_max = float(np.max(np.abs(zeta)))
            p_max = float(np.max(np.abs(data['psi'])))
            psi_d = data['psi']
            dx = float(data.get('dx', 6.0e6 / 512))
            dy = float(data.get('dy', 3.0e6 / 256))
            u_c = -(psi_d[2:, 1:-1] - psi_d[:-2, 1:-1]) / (2 * dy)
            v_c = (psi_d[1:-1, 2:] - psi_d[1:-1, :-2]) / (2 * dx)
            v_max = float(np.sqrt(u_c**2 + v_c**2).max())
            max_zeta = max(max_zeta, z_max)
            max_psi = max(max_psi, p_max)
            max_vel = max(max_vel, v_max)
            n_scanned += 1
            del data, u_c, v_c
            gc.collect()
        except Exception:
            pass

    if n_scanned == 0:
        print("  AVISO: No se pudo escanear archivos. Usando valores por defecto.")
        return

    GLOBAL_MAX_ZETA = max(max_zeta, VMAX_ZETA_DEFAULT)
    GLOBAL_MAX_PSI = max(max_psi, VMAX_PSI_DEFAULT)
    GLOBAL_MAX_VEL = max(max_vel, VMAX_VEL_DEFAULT)

    print("  Pass 1: {} archivos validos escaneados".format(n_scanned))
    print("  GLOBAL_MAX_ZETA = {:.4e}".format(GLOBAL_MAX_ZETA))
    print("  GLOBAL_MAX_PSI  = {:.4e}".format(GLOBAL_MAX_PSI))
    print("  GLOBAL_MAX_VEL  = {:.4e}".format(GLOBAL_MAX_VEL))


def _compute_fps(n_snapshots, target_seconds=60):
    fps = n_snapshots / target_seconds
    fps = max(1.0, min(60.0, fps))
    return fps


def _compute_velocity_field(psi, dx, dy):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dy)
    v[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx)
    return np.sqrt(u**2 + v**2)


def _find_nearest_snapshot(target_day, days, snapshot_files):
    """Encuentra el índice del snapshot con día más cercano a target_day."""
    days_arr = np.array(days)
    idx = np.argmin(np.abs(days_arr - target_day))
    return idx, snapshot_files[idx], days[idx]


def generate_snapshots(snapshot_files, days, modelos_dir, videos_dir,
                       vmax_zeta, vmax_psi, vmax_vel, interval_days=50,
                       all_modelos=None, nlist=None):
    """Genera imágenes PNG cada interval_days días para vorticidad, psi y velocidad.

    Las imágenes se guardan en subcarpetas dentro de videos_dir:
      videos_dir/vorticidad/snapshot_dia_{day}_vorticidad.png
      videos_dir/psi/snapshot_dia_{day}_psi.png
      videos_dir/velocidad/snapshot_dia_{day}_velocidad.png

    Si se proporciona all_modelos y nlist (modo comparativa), los snapshots
    con partículas muestran las 3 resoluciones con colores y leyenda.

    Se renderiza siempre a la máxima calidad (2K) independientemente de la
    selección de calidad de vídeo.
    """
    import matplotlib.pyplot as plt

    # Forzar calidad máxima (2K) para los snapshots
    snap_figsize = (12.8, 7.2)
    snap_dpi = 200
    snap_title_size = 22
    snap_label_size = 18
    snap_cbar_label_size = 18
    snap_cbar_tick_size = 14

    subdirs = {
        'vorticidad': os.path.join(videos_dir, 'vorticidad'),
        'psi': os.path.join(videos_dir, 'psi'),
        'velocidad': os.path.join(videos_dir, 'velocidad'),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    max_day = max(days)
    target_days = list(range(0, int(max_day) + 1, interval_days))

    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir, f))
        if first_metadata:
            break
    if first_metadata is None:
        print("  ERROR: No se pudo obtener metadatos del grid.")
        return

    x_km, y_km, X, Y = _compute_grid(first_metadata)
    dx = first_metadata.get('dx', 6.0e6 / 512)
    dy = first_metadata.get('dy', 3.0e6 / 256)

    es_comparativa = all_modelos is not None and nlist is not None

    print(f"\n  Generando snapshots 2K cada {interval_days} días ({len(target_days)} imágenes)...")

    for target_day in target_days:
        idx, fname, actual_day = _find_nearest_snapshot(target_day, days, snapshot_files)
        day_str = f"{actual_day:.0f}"

        try:
            data = np.load(os.path.join(modelos_dir, fname))
        except Exception:
            print(f"    AVISO: No se pudo cargar {fname}, saltando día {target_day}")
            continue

        zeta = data['zeta']
        psi = data['psi']
        vel = _compute_velocity_field(psi, dx, dy)
        px = data.get('particles_x', None)
        py = data.get('particles_y', None)
        data.close()

        def _render_field_only(field_data, cmap, vmin, vmax, cbar_label, contour_data, contour_kwargs,
                               title_str, subdir_key, base_name):
            fig, ax = plt.subplots(figsize=snap_figsize, dpi=snap_dpi)
            im = ax.pcolormesh(X, Y, field_data, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading='nearest')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, fontsize=snap_cbar_label_size)
            cbar.ax.tick_params(labelsize=snap_cbar_tick_size)
            if contour_data is not None:
                ax.contour(X, Y, contour_data, **contour_kwargs)
            ax.set_xlabel('Distancia [km]', fontsize=snap_label_size)
            ax.set_ylabel('Distancia [km]', fontsize=snap_label_size)
            ax.set_title(title_str, fontsize=snap_title_size, pad=8, fontweight='bold')
            ax.set_aspect('equal')
            fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(snap_cbar_tick_size - 3, 9),
                     color='gray', ha='left', alpha=0.7, transform=fig.transFigure)
            plt.tight_layout(rect=[0, 0.04, 1, 0.96])
            save_path = os.path.join(subdirs[subdir_key],
                                     f"snapshot_dia_{day_str}_{base_name}.png")
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=snap_dpi)
            print(f"      Guardado: {os.path.basename(save_path)}")
            plt.close(fig)
            plt.cla()
            plt.clf()
            gc.collect()

        def _render_with_particles(field_data, cmap, vmin, vmax, cbar_label, contour_data,
                                    contour_kwargs, title_str, subdir_key, base_name,
                                    sx, sy):
            fig, ax = plt.subplots(figsize=snap_figsize, dpi=snap_dpi)
            im = ax.pcolormesh(X, Y, field_data, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading='nearest')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, fontsize=snap_cbar_label_size)
            cbar.ax.tick_params(labelsize=snap_cbar_tick_size)
            if contour_data is not None:
                ax.contour(X, Y, contour_data, **contour_kwargs)
            ax.scatter(sx, sy, c='#B50000', s=4,
                      edgecolors='#5C0000', linewidths=0.3, alpha=0.8, zorder=5)
            ax.set_xlabel('Distancia [km]', fontsize=snap_label_size)
            ax.set_ylabel('Distancia [km]', fontsize=snap_label_size)
            ax.set_title(title_str, fontsize=snap_title_size, pad=8, fontweight='bold')
            ax.set_aspect('equal')
            fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(snap_cbar_tick_size - 3, 9),
                     color='gray', ha='left', alpha=0.7, transform=fig.transFigure)
            plt.tight_layout(rect=[0, 0.04, 1, 0.96])
            save_path = os.path.join(subdirs[subdir_key],
                                     f"snapshot_dia_{day_str}_{base_name}.png")
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=snap_dpi)
            print(f"      Guardado: {os.path.basename(save_path)}")
            plt.close(fig)
            plt.cla()
            plt.clf()
            gc.collect()

        if es_comparativa:
            # Modo comparativa: cargar partículas de todas las resoluciones
            comp_particles = _load_particle_data(all_modelos, fname)

            def _render_comparison_with_particles(field_data, cmap, vmin, vmax, cbar_label,
                                                   contour_data, contour_kwargs, title_str,
                                                   subdir_key, base_name):
                fig, ax = plt.subplots(figsize=snap_figsize, dpi=snap_dpi)
                im = ax.pcolormesh(X, Y, field_data, cmap=cmap,
                                  vmin=vmin, vmax=vmax, shading='nearest')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(cbar_label, fontsize=snap_cbar_label_size)
                cbar.ax.tick_params(labelsize=snap_cbar_tick_size)
                if contour_data is not None:
                    ax.contour(X, Y, contour_data, **contour_kwargs)
                # Partículas de cada resolución con su color
                for n_val in sorted(comp_particles.keys()):
                    cpx, cpy = comp_particles[n_val]
                    color = COMPARISON_COLORS.get(n_val, '#B50000')
                    if len(cpx) > 0:
                        ax.scatter(cpx, cpy, c=color, s=4, edgecolors='none',
                                  alpha=0.8, zorder=5,
                                  label=COMPARISON_LABELS.get(n_val, f'N={n_val}'))
                ax.legend(fontsize=snap_cbar_tick_size, loc='upper right', markerscale=3)
                ax.set_xlabel('Distancia [km]', fontsize=snap_label_size)
                ax.set_ylabel('Distancia [km]', fontsize=snap_label_size)
                ax.set_title(title_str, fontsize=snap_title_size, pad=8, fontweight='bold')
                ax.set_aspect('equal')
                fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(snap_cbar_tick_size - 3, 9),
                         color='gray', ha='left', alpha=0.7, transform=fig.transFigure)
                plt.tight_layout(rect=[0, 0.04, 1, 0.96])
                save_path = os.path.join(subdirs[subdir_key],
                                         f"snapshot_dia_{day_str}_{base_name}.png")
                plt.savefig(save_path, format='png', bbox_inches='tight', dpi=snap_dpi)
                print(f"      Guardado: {os.path.basename(save_path)}")
                plt.close(fig)
                plt.cla()
                plt.clf()
                gc.collect()

            # Vorticidad — comparativa
            _render_field_only(zeta, 'RdBu_r', -vmax_zeta, vmax_zeta,
                               r'Vorticidad $\zeta$ [s$^{-1}$]', psi,
                               dict(colors='#2d2d2d', linewidths=1.2, alpha=0.5, levels=15),
                               f"Vorticidad - Comparativa - Día {actual_day:.1f}", 'vorticidad',
                               'vorticidad_sin_particulas')
            _render_comparison_with_particles(zeta, 'RdBu_r', -vmax_zeta, vmax_zeta,
                                              r'Vorticidad $\zeta$ [s$^{-1}$]', psi,
                                              dict(colors='#2d2d2d', linewidths=1.2, alpha=0.5, levels=15),
                                              f"Vorticidad - Comparativa - Día {actual_day:.1f}", 'vorticidad',
                                              'vorticidad_con_particulas')

            # Función de corriente — comparativa
            _render_field_only(psi, 'coolwarm', -vmax_psi, vmax_psi,
                               r'$\psi$ [m$^2$/s]', psi,
                               dict(colors='k', linewidths=0.9, alpha=0.45, levels=15),
                               f"Función de corriente - Comparativa - Día {actual_day:.1f}", 'psi',
                               'psi_sin_particulas')
            _render_comparison_with_particles(psi, 'coolwarm', -vmax_psi, vmax_psi,
                                              r'$\psi$ [m$^2$/s]', psi,
                                              dict(colors='k', linewidths=0.9, alpha=0.45, levels=15),
                                              f"Función de corriente - Comparativa - Día {actual_day:.1f}", 'psi',
                                              'psi_con_particulas')

            # Velocidad — comparativa
            _render_field_only(vel, 'YlGnBu', 0.0, vmax_vel,
                               'Velocidad [m/s]', None, {},
                               f"Velocidad - Comparativa - Día {actual_day:.1f}", 'velocidad',
                               'velocidad_sin_particulas')
            _render_comparison_with_particles(vel, 'YlGnBu', 0.0, vmax_vel,
                                              'Velocidad [m/s]', None, {},
                                              f"Velocidad - Comparativa - Día {actual_day:.1f}", 'velocidad',
                                              'velocidad_con_particulas')
        else:
            # Modo monoejecución: partículas del archivo local
            has_p = px is not None and py is not None and len(px) > 0
            sx, sy = (px / 1e3, py / 1e3) if has_p else (None, None)

            # Vorticidad — sin y con partículas
            _render_field_only(zeta, 'RdBu_r', -vmax_zeta, vmax_zeta,
                               r'Vorticidad $\zeta$ [s$^{-1}$]', psi,
                               dict(colors='#2d2d2d', linewidths=1.2, alpha=0.5, levels=15),
                               f"Vorticidad - Día {actual_day:.1f}", 'vorticidad',
                               'vorticidad_sin_particulas')
            if has_p:
                _render_with_particles(zeta, 'RdBu_r', -vmax_zeta, vmax_zeta,
                                       r'Vorticidad $\zeta$ [s$^{-1}$]', psi,
                                       dict(colors='#2d2d2d', linewidths=1.2, alpha=0.5, levels=15),
                                       f"Vorticidad - Día {actual_day:.1f}", 'vorticidad',
                                       'vorticidad_con_particulas', sx, sy)

            # Función de corriente — sin y con partículas
            _render_field_only(psi, 'coolwarm', -vmax_psi, vmax_psi,
                               r'$\psi$ [m$^2$/s]', psi,
                               dict(colors='k', linewidths=0.9, alpha=0.45, levels=15),
                               f"Función de corriente - Día {actual_day:.1f}", 'psi',
                               'psi_sin_particulas')
            if has_p:
                _render_with_particles(psi, 'coolwarm', -vmax_psi, vmax_psi,
                                       r'$\psi$ [m$^2$/s]', psi,
                                       dict(colors='k', linewidths=0.9, alpha=0.45, levels=15),
                                       f"Función de corriente - Día {actual_day:.1f}", 'psi',
                                       'psi_con_particulas', sx, sy)

            # Velocidad — sin y con partículas
            _render_field_only(vel, 'YlGnBu', 0.0, vmax_vel,
                               'Velocidad [m/s]', None, {},
                               f"Velocidad - Día {actual_day:.1f}", 'velocidad',
                               'velocidad_sin_particulas')
            if has_p:
                _render_with_particles(vel, 'YlGnBu', 0.0, vmax_vel,
                                       'Velocidad [m/s]', None, {},
                                       f"Velocidad - Día {actual_day:.1f}", 'velocidad',
                                       'velocidad_con_particulas', sx, sy)

        print(f"    Snapshots día {actual_day:.0f} generados (vorticidad, psi, velocidad)")

    print(f"  [OK] {len(target_days)} snapshots 2K generados en {videos_dir}")


def save_fields_animation(snapshot_files, t_f, modelos_dir, save_p, vmax_zeta, vmax_psi, fps):
    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir, f))
        if first_metadata:
            break
    x, y, X, Y = _compute_grid(first_metadata)
    n = len(snapshot_files)

    frames = []
    for i in range(n):
        try:
            data = np.load(os.path.join(modelos_dir, snapshot_files[i]))
            px = data.get('particles_x', None)
            py = data.get('particles_y', None)
            frames.append((data['zeta'], data['psi'], t_f[i], i, px, py))
            del data
            gc.collect()
        except Exception:
            pass

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI_SAVE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=CBAR_TICK_SIZE)
    ax.set_aspect('equal')

    vmax_z = vmax_zeta
    im = ax.pcolormesh(X, Y, frames[0][0], cmap='RdBu_r',
                      vmin=-vmax_z, vmax=vmax_z, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Vorticidad $\zeta$ [s$^{-1}$]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(CBAR_TICK_SIZE)
    cset = ax.contour(X, Y, frames[0][1], colors='#2d2d2d',
                     linewidths=1.2, alpha=0.5, levels=15)
    scatter_pts = ax.scatter([], [], c='#B50000', s=4, edgecolors='#5C0000',
                            linewidths=0.3, alpha=0.8, zorder=5)
    title_artist = ax.set_title(fr"Día 0.00", fontsize=VIDEO_TITLE_SIZE, pad=8, fontweight='bold')
    fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(CBAR_TICK_SIZE - 3, 9), color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    save_p_sin = save_p.replace('.mp4', '_sin_particulas.mp4')
    save_p_con = save_p.replace('.mp4', '_con_particulas.mp4')

    def _render_pass(has_particles, out_path, desc_suffix):
        nonlocal cset
        writer, encoder_name = _create_writer(fps)
        pbar = tqdm(total=len(frames), desc=f"Vídeo: Vorticidad {desc_suffix}", position=0, leave=True)
        success = False
        try:
            with writer.saving(fig, out_path, dpi=DPI_SAVE):
                for idx in range(len(frames)):
                    if idx % FRAME_SKIP != 0:
                        pbar.update(1)
                        continue
                    zeta, psi, day, _, px, py = frames[idx]
                    if np.isnan(zeta).any() or np.isnan(psi).any():
                        pbar.update(FRAME_SKIP)
                        continue
                    if np.max(np.abs(zeta)) < 1e-18 or np.max(np.abs(psi)) < 1e-18:
                        pbar.update(FRAME_SKIP)
                        continue
                    im.set_array(zeta.ravel())
                    if cset is not None:
                        cset.remove()
                    cset = ax.contour(X, Y, psi, colors='#2d2d2d', linewidths=1.2,
                                     alpha=0.5, levels=np.linspace(-vmax_psi, vmax_psi, 15),
                                     linestyles='solid')
                    if has_particles and px is not None and py is not None and len(px) > 0:
                        scatter_pts.set_offsets(np.column_stack((px / 1e3, py / 1e3)))
                    else:
                        scatter_pts.set_offsets(np.empty((0, 2)))
                    title_artist.set_text(fr"Vorticidad - Día {day:.1f}")
                    writer.grab_frame()
                    pbar.update(FRAME_SKIP)
            success = True
        except Exception as e:
            print(f"\n[ERROR] Vídeo vorticidad {desc_suffix} ({encoder_name}): {e}")
        finally:
            pbar.close()
        if success:
            print(f"\n[OK] Vídeo vorticidad {desc_suffix} ({encoder_name}) -> {out_path}")
        sys.stdout.flush()

    _render_pass(False, save_p_sin, '(sin partículas)')
    _render_pass(True, save_p_con, '(con partículas)')

    plt.close(fig)
    plt.cla()
    plt.clf()
    gc.collect()


def save_psi_animation(snapshot_files, t_f, modelos_dir, save_p, vmax_psi, fps):
    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir, f))
        if first_metadata:
            break
    x, y, X, Y = _compute_grid(first_metadata)
    n = len(snapshot_files)

    frames = []
    for i in range(n):
        try:
            data = np.load(os.path.join(modelos_dir, snapshot_files[i]))
            px = data.get('particles_x', None)
            py = data.get('particles_y', None)
            frames.append((data['psi'], t_f[i], px, py))
            del data
            gc.collect()
        except Exception:
            pass

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI_SAVE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=CBAR_TICK_SIZE)
    ax.set_aspect('equal')

    vmax_p = vmax_psi
    im = ax.pcolormesh(X, Y, frames[0][0], cmap='coolwarm',
                      vmin=-vmax_p, vmax=vmax_p, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\psi$ [m$^2$/s]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(CBAR_TICK_SIZE)
    cset = ax.contour(X, Y, frames[0][0], colors='k', linewidths=0.9,
                     alpha=0.45, levels=15)
    scatter_pts = ax.scatter([], [], c='#B50000', s=4, edgecolors='#5C0000',
                            linewidths=0.3, alpha=0.8, zorder=5)
    title_artist = ax.set_title(fr"Día 0.00", fontsize=VIDEO_TITLE_SIZE, pad=8, fontweight='bold')
    fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(CBAR_TICK_SIZE - 3, 9), color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    save_p_sin = save_p.replace('.mp4', '_sin_particulas.mp4')
    save_p_con = save_p.replace('.mp4', '_con_particulas.mp4')

    def _render_pass(has_particles, out_path, desc_suffix):
        nonlocal cset
        writer, encoder_name = _create_writer(fps)
        pbar = tqdm(total=len(frames), desc=f"Vídeo: Función de Corriente {desc_suffix}", position=0, leave=True)
        success = False
        try:
            with writer.saving(fig, out_path, dpi=DPI_SAVE):
                for idx in range(len(frames)):
                    if idx % FRAME_SKIP != 0:
                        pbar.update(1)
                        continue
                    psi, day, px, py = frames[idx]
                    if np.isnan(psi).any():
                        pbar.update(FRAME_SKIP)
                        continue
                    if np.max(np.abs(psi)) < 1e-18:
                        pbar.update(FRAME_SKIP)
                        continue
                    im.set_array(psi.ravel())
                    if cset is not None:
                        cset.remove()
                    cset = ax.contour(X, Y, psi, colors='k', linewidths=0.9,
                                     alpha=0.45, levels=np.linspace(-vmax_p, vmax_p, 15),
                                     linestyles='solid')
                    if has_particles and px is not None and py is not None and len(px) > 0:
                        scatter_pts.set_offsets(np.column_stack((px / 1e3, py / 1e3)))
                    else:
                        scatter_pts.set_offsets(np.empty((0, 2)))
                    title_artist.set_text(fr"Función de Corriente - Día {day:.1f}")
                    writer.grab_frame()
                    pbar.update(FRAME_SKIP)
            success = True
        except Exception as e:
            print(f"\n[ERROR] Vídeo streamfunction {desc_suffix} ({encoder_name}): {e}")
        finally:
            pbar.close()
        if success:
            print(f"\n[OK] Vídeo streamfunction {desc_suffix} ({encoder_name}) -> {out_path}")
        sys.stdout.flush()

    _render_pass(False, save_p_sin, '(sin partículas)')
    _render_pass(True, save_p_con, '(con partículas)')

    plt.close(fig)
    plt.cla()
    plt.clf()
    gc.collect()


def save_velocity_animation(snapshot_files, t_f, modelos_dir, save_p, vmax_vel, fps):
    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir, f))
        if first_metadata:
            break
    x, y, X, Y = _compute_grid(first_metadata)
    n = len(snapshot_files)

    frames = []
    metadata = first_metadata or {}
    dx = metadata.get('dx', 6.0e6 / 512)
    dy = metadata.get('dy', 3.0e6 / 256)
    for i in range(n):
        try:
            data = np.load(os.path.join(modelos_dir, snapshot_files[i]))
            vel = _compute_velocity_field(data['psi'], dx, dy)
            px = data.get('particles_x', None)
            py = data.get('particles_y', None)
            frames.append((vel, t_f[i], px, py))
            del data
            gc.collect()
        except Exception:
            pass

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI_SAVE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=CBAR_TICK_SIZE)
    ax.set_aspect('equal')

    vmax_v = vmax_vel
    im = ax.pcolormesh(X, Y, frames[0][0], cmap='YlGnBu',
                      vmin=0.0, vmax=vmax_v, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Velocidad [m/s]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(CBAR_TICK_SIZE)
    scatter_pts = ax.scatter([], [], c='#B50000', s=4, edgecolors='#5C0000',
                            linewidths=0.3, alpha=0.8, zorder=5)
    title_artist = ax.set_title(fr"Día 0.00", fontsize=VIDEO_TITLE_SIZE, pad=8, fontweight='bold')
    fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(CBAR_TICK_SIZE - 3, 9), color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    save_p_sin = save_p.replace('.mp4', '_sin_particulas.mp4')
    save_p_con = save_p.replace('.mp4', '_con_particulas.mp4')

    def _render_pass(has_particles, out_path, desc_suffix):
        writer, encoder_name = _create_writer(fps)
        pbar = tqdm(total=len(frames), desc=f"Vídeo: Velocidad {desc_suffix}", position=0, leave=True)
        success = False
        try:
            with writer.saving(fig, out_path, dpi=DPI_SAVE):
                for idx in range(len(frames)):
                    if idx % FRAME_SKIP != 0:
                        pbar.update(1)
                        continue
                    vel, day, px, py = frames[idx]
                    if np.isnan(vel).any():
                        pbar.update(FRAME_SKIP)
                        continue
                    if np.max(np.abs(vel)) < 1e-18:
                        pbar.update(FRAME_SKIP)
                        continue
                    im.set_array(vel.ravel())
                    if has_particles and px is not None and py is not None and len(px) > 0:
                        scatter_pts.set_offsets(np.column_stack((px / 1e3, py / 1e3)))
                    else:
                        scatter_pts.set_offsets(np.empty((0, 2)))
                    title_artist.set_text(fr"Módulo de Velocidad - Día {day:.1f}")
                    writer.grab_frame()
                    pbar.update(FRAME_SKIP)
            success = True
        except Exception as e:
            print(f"\n[ERROR] Vídeo velocidad {desc_suffix} ({encoder_name}): {e}")
        finally:
            pbar.close()
        if success:
            print(f"\n[OK] Vídeo velocidad {desc_suffix} ({encoder_name}) -> {out_path}")
        sys.stdout.flush()

    _render_pass(False, save_p_sin, '(sin partículas)')
    _render_pass(True, save_p_con, '(con partículas)')

    plt.close(fig)
    plt.cla()
    plt.clf()
    gc.collect()


def post_process():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    modelos_dir = os.path.join(output_dir, "modelos")
    videos_dir = os.path.join(output_dir, "videos")
    metricas_dir = os.path.join(output_dir, "metricas")

    for d in [videos_dir, metricas_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("POST-PROCESADO - Generando visualizaciones (60s, flicker-free, {})".format(QUALITY))
    print("=" * 60)

    # Detectar modo comparativa (múltiples resoluciones)
    nlist = _detect_available_counts(output_dir)
    if len(nlist) >= 2:
        print("\n  Detectadas {} simulaciones con diferente resolución:".format(len(nlist)))
        for n in nlist:
            print(f"    - N={n} partículas")
        print("  Generando POST-PROCESADO COMPARATIVO (las 3 resoluciones simultáneamente)")
        _run_comparison_post_process(output_dir, videos_dir, metricas_dir, nlist)
        return

    # Modo tradicional (una sola simulación)
    files = [f for f in os.listdir(modelos_dir) if f.endswith('.npz')]
    if not files:
        print("Error: No se encontraron archivos .npz. Ejecuta primero main.py.")
        return

    day_groups = {}
    for f in files:
        try:
            data = np.load(os.path.join(modelos_dir, f))
            day = float(data.get('t_days', data.get('t', 0) / 86400.0))
            mag = float(np.max(np.abs(data['zeta'])))
            del data
            gc.collect()
            rounded = round(day, 1)
            if rounded not in day_groups:
                day_groups[rounded] = []
            day_groups[rounded].append((f, day, mag))
        except Exception:
            pass

    for rounded in day_groups:
        day_groups[rounded].sort(key=lambda x: x[1])

    snapshot_files = []
    days = []
    discarded = 0
    for rounded in sorted(day_groups.keys()):
        candidates = day_groups[rounded]
        best_file = None
        best_day = None

        for f, day, mag in candidates:
            if mag > 1e-18:
                if best_file is None or day > best_day:
                    best_file = f
                    best_day = day

        if best_file is not None:
            snapshot_files.append(best_file)
            days.append(best_day)
        else:
            discarded += 1
            print("  Descartando día {:.1f} (sin datos válidos)".format(rounded))

    n_snapshots = len(snapshot_files)
    print("  Snapshots válidos: {} | Días descartados: {} | Rango: {:.2f} - {:.2f}".format(
        n_snapshots, discarded, days[0] if days else 0, days[-1] if days else 0))

    fps = n_snapshots / 60.0
    print("  FPS: {:.2f} (duración: 60.0s)".format(fps))

    if RUN_VIDEOS or RUN_SNAPSHOTS:
        _scan_global_maxima(modelos_dir, snapshot_files, sample_rate=5)

    print("\n--- Generando visualizaciones ---")

    if RUN_PLOTS:
        print("\n[GRÁFICAS] Generando gráficas métricas en formato vectorial (SVG)...")
        plot_energy_history()
        print("[OK] Gráficas métricas (SVG) generadas en la carpeta output.")

        print("\n[GRÁFICAS] Generando gráficas de métricas de dispersión...")
        plot_dispersion_metrics()
        print("[OK] Gráficas de dispersión generadas.")

        print("\n  Generando perfiles de viento...")
        _p = PhysicsConfig()
        plot_wind_stress_profile(_p.ly, _p.tau0,
                                 save_path=os.path.join(metricas_dir, "wind_stress_profile.svg"))
        plot_wind_stress_profile(_p.ly, _p.tau0,
                                 save_path=os.path.join(metricas_dir, "wind_stress_profile_centered.svg"),
                                 centered=True)
        plot_wind_curl_profile(_p.ly, _p.tau0,
                                save_path=os.path.join(metricas_dir, "wind_curl_profile.svg"))
        plot_wind_curl_profile(_p.ly, _p.tau0,
                                save_path=os.path.join(metricas_dir, "wind_curl_profile_centered.svg"),
                                centered=True)
        print("[OK] Perfiles de viento (SVG) generados en la carpeta output.")

    if RUN_VIDEOS:
        print("\n--- Generando 3 vídeos (secuencial, sin flicker) ---")

        fields_path = os.path.join(videos_dir, "fields_evolution_{}.mp4".format(QUALITY))
        psi_path = os.path.join(videos_dir, "psi_evolution_{}.mp4".format(QUALITY))
        vel_path = os.path.join(videos_dir, "velocity_evolution_{}.mp4".format(QUALITY))

        print("\n[VÍDEO 1/3] Generando vídeo de vorticidad...")
        save_fields_animation(snapshot_files, days, modelos_dir, fields_path,
                              GLOBAL_MAX_ZETA, GLOBAL_MAX_PSI, fps)

        print("\n[VÍDEO 2/3] Generando vídeo de función de corriente...")
        save_psi_animation(snapshot_files, days, modelos_dir, psi_path,
                           GLOBAL_MAX_PSI, fps)

        print("\n[VÍDEO 3/3] Generando vídeo de módulo de velocidad...")
        save_velocity_animation(snapshot_files, days, modelos_dir, vel_path,
                                GLOBAL_MAX_VEL, fps)

    if RUN_SNAPSHOTS:
        print("\n[SNAPSHOTS] Generando snapshots estáticos (2K) cada 50 días...")
        generate_snapshots(snapshot_files, days, modelos_dir, videos_dir,
                           GLOBAL_MAX_ZETA, GLOBAL_MAX_PSI, GLOBAL_MAX_VEL)
        print("[OK]")

    plt.close('all')
    gc.collect()

    print("=" * 60)
    print("POST-PROCESADO COMPLETO")
    print("=" * 60)


def _run_comparison_post_process(output_dir, videos_dir, metricas_dir, nlist):
    """Post-procesado comparativo de simulaciones con distinto número de partículas."""
    from config import PhysicsConfig

    # Usar la resolución con más partículas para los snapshots de campos
    base_n = max(nlist)
    suffix = f"_res_{base_n}"
    base_modelos = _get_modelos_dir(output_dir, suffix)
    all_modelos = {n: _get_modelos_dir(output_dir, f"_res_{n}") for n in nlist}

    files = sorted([f for f in os.listdir(base_modelos) if f.endswith('.npz')])
    if not files:
        print("Error: No se encontraron snapshots en", base_modelos)
        return

    # Extraer días y snapshots válidos
    snapshot_files = []
    days = []
    for f in files:
        try:
            data = np.load(os.path.join(base_modelos, f))
            day = float(data.get('t_days', data.get('t', 0) / 86400.0))
            mag = float(np.max(np.abs(data['zeta'])))
            del data
            gc.collect()
            if mag > 1e-18:
                snapshot_files.append(f)
                days.append(day)
        except Exception:
            pass

    n_snapshots = len(snapshot_files)
    fps = n_snapshots / 60.0
    print("  Snapshots: {} | FPS: {:.2f}".format(n_snapshots, fps))

    # Escanear máximos globales desde la resolución base
    _scan_global_maxima(base_modelos, snapshot_files, sample_rate=5)

    print("\n--- Generando visualizaciones COMPARATIVAS ---")

    if RUN_PLOTS:
        # Gráficas comparativas de dispersión
        print("\n[GRÁFICAS] Gráficas comparativas de dispersión...")
        plot_comparison_dispersion_metrics(output_dir, metricas_dir, nlist)
        print("[OK]")

        # Gráficas comparativas de energía (un solo archivo simulation_summary.npz)
        print("\n[GRÁFICAS] Gráficas comparativas de energía y enstrofía...")
        plot_comparison_energy_history(output_dir, metricas_dir)
        print("[OK]")

        # Perfiles de viento (no cambian con resolución)
        print("\n  Generando perfiles de viento...")
        _p = PhysicsConfig()
        plot_wind_stress_profile(_p.ly, _p.tau0,
                                 save_path=os.path.join(metricas_dir, "wind_stress_profile.svg"))
        plot_wind_stress_profile(_p.ly, _p.tau0,
                                 save_path=os.path.join(metricas_dir, "wind_stress_profile_centered.svg"),
                                 centered=True)
        plot_wind_curl_profile(_p.ly, _p.tau0,
                                save_path=os.path.join(metricas_dir, "wind_curl_profile.svg"))
        plot_wind_curl_profile(_p.ly, _p.tau0,
                                save_path=os.path.join(metricas_dir, "wind_curl_profile_centered.svg"),
                                centered=True)
        print("[OK]")

    if RUN_VIDEOS:
        # Vídeos comparativos (partículas de las 3 resoluciones sobre el mismo campo)
        print("\n[VÍDEO 1/3] Vídeo comparativo de vorticidad...")
        fields_path = os.path.join(videos_dir, "fields_comparison_{}.mp4".format(QUALITY))
        save_comparison_fields_animation(snapshot_files, days, all_modelos, fields_path,
                                          GLOBAL_MAX_ZETA, GLOBAL_MAX_PSI, fps)
        print("[OK]")

        print("\n[VÍDEO 2/3] Vídeo comparativo de función de corriente...")
        psi_path = os.path.join(videos_dir, "psi_comparison_{}.mp4".format(QUALITY))
        save_comparison_psi_animation(snapshot_files, days, all_modelos, psi_path,
                                       GLOBAL_MAX_PSI, fps)
        print("[OK]")

        # Vídeo comparativo de velocidad
        vel_path = os.path.join(videos_dir, "velocity_comparison_{}.mp4".format(QUALITY))
        print(f"\n[VÍDEO 3/3] Vídeo comparativo de velocidad ({QUALITY})...")
        save_comparison_velocity_animation(snapshot_files, days, all_modelos, vel_path,
                                           GLOBAL_MAX_VEL, fps, nlist)
        print("[OK]")

    if RUN_SNAPSHOTS:
        # Snapshots estáticos cada 50 días
        print("\n[SNAPSHOTS] Generando snapshots estáticos (2K) cada 50 días...")
        generate_snapshots(snapshot_files, days, base_modelos, videos_dir,
                           GLOBAL_MAX_ZETA, GLOBAL_MAX_PSI, GLOBAL_MAX_VEL,
                           all_modelos=all_modelos, nlist=nlist)
        print("[OK]")

    plt.close('all')
    gc.collect()

    print("=" * 60)
    print("POST-PROCESADO COMPARATIVO COMPLETO")
    print("=" * 60)


def plot_energy_history():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    metricas_dir = os.path.join(output_dir, "metricas")
    os.makedirs(metricas_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "simulation_summary.npz")
    if not os.path.exists(summary_path):
        print("Error: simulation_summary.npz no encontrado. Ejecuta main.py primero.")
        return

    data = np.load(summary_path)
    t_days = data['energy_days']
    energy_vals = data['energy']
    enstrophy_vals = data['enstrophy']

    paper_fonts = {
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
        'figure.titlesize': 16,
    }
    with plt.rc_context(paper_fonts):
        fig, ax1 = plt.subplots(figsize=(11.5, 5.8))

        color_e = '#2E86AB'
        color_z = '#E94F37'

        l1 = ax1.plot(t_days, energy_vals, color=color_e, lw=2.0, alpha=0.85, label='Energía')
        ax1.fill_between(t_days, energy_vals, alpha=0.1, color=color_e)
        ax1.set_xlabel('Tiempo [días]')
        ax1.set_ylabel('Energía cinética total [J/kg]', color=color_e)
        ax1.tick_params(axis='y', labelcolor=color_e)
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        l2 = ax2.plot(t_days, enstrophy_vals, color=color_z, lw=2.0, alpha=0.85, label='Enstrofía')
        ax2.fill_between(t_days, enstrophy_vals, alpha=0.1, color=color_z)
        ax2.set_ylabel('Enstrofía total [s⁻²]', color=color_z)
        ax2.tick_params(axis='y', labelcolor=color_z)
        ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))

        mask = (t_days >= 500) & (t_days <= 2000)
        if np.count_nonzero(mask) >= 2:
            energy_mean = np.mean(energy_vals[mask])
            enstrophy_mean = np.mean(enstrophy_vals[mask])
            ax1.axhline(energy_mean, color=color_e, linestyle='--', linewidth=1.6, alpha=0.65, label='Media energía 500–2000 días')
            ax2.axhline(enstrophy_mean, color=color_z, linestyle='--', linewidth=1.6, alpha=0.65, label='Media enstrofía 500–2000 días')
        else:
            print("[WARN] No hay suficientes datos entre 500–2000 días; se omiten líneas de tendencia.")

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        # añadir líneas de media a la leyenda si existen
        handles = [l1[0], l2[0]]
        hlabel = ['Energía', 'Enstrofía']
        if np.count_nonzero(mask) >= 2:
            from matplotlib.lines import Line2D
            h_energy_mean = Line2D([], [], color=color_e, ls='--', lw=1.6, alpha=0.65, label='Media energía 500–2000 días')
            h_enstrophy_mean = Line2D([], [], color=color_z, ls='--', lw=1.6, alpha=0.65, label='Media enstrofía 500–2000 días')
            handles = [l1[0], h_energy_mean, l2[0], h_enstrophy_mean]
        ax1.legend(handles=handles, loc='upper right')

        ax1.set_title('Evolución de energía y enstrofía', fontweight='bold', pad=15)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        fig.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                 ha='left', alpha=0.7, transform=fig.transFigure)

        save_path = os.path.join(metricas_dir, "energy_evolution.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close('all')
        gc.collect()
        print(f"  Guardado: energy_evolution.svg")

    data.close()
    print("\nEnergía y Enstrofía en: {}".format(metricas_dir))


def plot_dispersion_metrics():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    metricas_dir = os.path.join(output_dir, "metricas")
    os.makedirs(metricas_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "simulation_summary.npz")
    if not os.path.exists(summary_path):
        print("Error: simulation_summary.npz no encontrado.")
        return

    data = np.load(summary_path)
    area_km2 = data.get('area_km2', None)
    rms_disp = data.get('rms_dispersion', None)
    centroids_arr = data.get('centroids', None)
    metrics_days = data.get('metrics_days', None)
    release_day = float(data.get('release_day', 0.0))
    lx = float(data.get('lx', 6.0e6))
    ly = float(data.get('ly', 3.0e6))

    if area_km2 is None or len(area_km2) == 0:
        print("  No hay datos de dispersión. Omitiendo gráficas de partículas.")
        data.close()
        return

    t_since_release = metrics_days - release_day
    paper_fonts = {
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
        'figure.titlesize': 16,
    }

    with plt.rc_context(paper_fonts):
        fig1, ax1 = plt.subplots(figsize=(11.5, 5.8))
        ax1.plot(t_since_release, area_km2, color='#d62728', lw=2.5, marker='o', ms=5)
        ax1.fill_between(t_since_release, area_km2, alpha=0.15, color='#d62728')
        ax1.set_xlabel('Tiempo desde el accidente [días]')
        ax1.set_ylabel('Área de búsqueda [km²]', color='#d62728')
        ax1.set_title('Evolución del área de búsqueda', fontweight='bold', pad=12)
        ax1.ticklabel_format(style='plain', axis='y')
        ax1.grid(True, alpha=0.3)
        fig1.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                  ha='left', alpha=0.7, transform=fig1.transFigure)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fp = os.path.join(metricas_dir, "search_area_evolution.svg")
        plt.savefig(fp, format='svg', bbox_inches='tight')
        print(f"  Guardado: search_area_evolution.svg")
        plt.close(fig1)

        rms_disp_km = rms_disp / 1e3
        fig2, ax2 = plt.subplots(figsize=(11.5, 5.8))
        ax2.loglog(t_since_release, rms_disp_km, color='#1f77b4', lw=2.5, marker='s', ms=5)
        ax2.set_xlabel('Tiempo desde el accidente [días]')
        ax2.set_ylabel('Radio RMS [km]', color='#1f77b4')
        ax2.set_title('Dispersión RMS (Escala Log-Log)', fontweight='bold', pad=12)
        ax2.grid(True, alpha=0.3, which='both')
        fig2.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                  ha='left', alpha=0.7, transform=fig2.transFigure)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fp = os.path.join(metricas_dir, "rms_dispersion_loglog.svg")
        plt.savefig(fp, format='svg', bbox_inches='tight')
        print(f"  Guardado: rms_dispersion_loglog.svg")
        plt.close(fig2)

        if centroids_arr is not None and centroids_arr.ndim == 2 and centroids_arr.shape[1] == 2:
            cx_km = centroids_arr[:, 0] / 1e3
            cy_km = centroids_arr[:, 1] / 1e3
            fig3, ax3 = plt.subplots(figsize=(11.5, 5.8))
            ax3.plot(cx_km, cy_km, color='#2ca02c', lw=2.5, marker='o', markersize=6)
            ax3.scatter(cx_km[0], cy_km[0], c='green', s=80, marker='^',
                        edgecolors='black', zorder=5, label='Inicio')
            ax3.scatter(cx_km[-1], cy_km[-1], c='red', s=80, marker='v',
                        edgecolors='black', zorder=5, label='Final')
            ax3.set_xlabel('X [km]')
            ax3.set_ylabel('Y [km]')
            ax3.set_title('Trayectoria del centroide de la nube', fontweight='bold', pad=12)
            ax3.set_xlim(0, lx / 1e3)
            ax3.set_ylim(0, ly / 1e3)
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            fig3.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                      ha='left', alpha=0.7, transform=fig3.transFigure)
            plt.tight_layout(rect=[0, 0.06, 1, 0.95])
            fp = os.path.join(metricas_dir, "centroid_trajectory.svg")
            plt.savefig(fp, format='svg', bbox_inches='tight')
            print(f"  Guardado: centroid_trajectory.svg")
            plt.close(fig3)

    data.close()
    print("\nMétricas de dispersión en: {}".format(metricas_dir))


def _load_particle_data(all_modelos, filename):
    """Carga partículas desde varios directorios para un mismo snapshot."""
    particles = {}
    for suf, dirpath in all_modelos.items():
        filepath = os.path.join(dirpath, filename)
        try:
            data = np.load(filepath)
            px = data.get('particles_x', None)
            py = data.get('particles_y', None)
            if px is not None and py is not None and len(px) > 0:
                particles[suf] = (px / 1e3, py / 1e3)
            data.close()
        except Exception:
            particles[suf] = (np.empty(0), np.empty(0))
    return particles


def save_comparison_fields_animation(snapshot_files, t_f, all_modelos, save_p, vmax_zeta, vmax_psi, fps):
    """Vídeo comparativo: campo de vorticidad + partículas de 3 resoluciones."""
    modelos_dir_base = list(all_modelos.values())[-1]
    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir_base, f))
        if first_metadata:
            break
    x, y, X, Y = _compute_grid(first_metadata)
    n = len(snapshot_files)

    frames = []
    for i in range(n):
        try:
            data = np.load(os.path.join(modelos_dir_base, snapshot_files[i]))
            frames.append((data['zeta'], data['psi'], t_f[i], i))
            del data
            gc.collect()
        except Exception:
            pass

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI_SAVE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=CBAR_TICK_SIZE)
    ax.set_aspect('equal')

    vmax_z = vmax_zeta
    im = ax.pcolormesh(X, Y, frames[0][0], cmap='RdBu_r',
                      vmin=-vmax_z, vmax=vmax_z, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Vorticidad $\zeta$ [s$^{-1}$]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(CBAR_TICK_SIZE)

    cset = ax.contour(X, Y, frames[0][1], colors='#2d2d2d',
                     linewidths=1.2, alpha=0.5, levels=15)

    # Tres grupos de partículas con distintos colores
    scatter_groups = {}
    for n in all_modelos:
        color = COMPARISON_COLORS.get(n, '#B50000')
        sc = ax.scatter([], [], c=color, s=4, edgecolors='none',
                        alpha=0.8, zorder=5, label=COMPARISON_LABELS.get(n, f'N={n}'))
        scatter_groups[n] = sc

    leg = ax.legend(fontsize=CBAR_TICK_SIZE, loc='upper right', markerscale=3)

    title_artist = ax.set_title(r"Día 0.00", fontsize=VIDEO_TITLE_SIZE, pad=8, fontweight='bold')
    fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(CBAR_TICK_SIZE - 3, 9), color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    save_p_sin = save_p.replace('.mp4', '_sin_particulas.mp4')
    save_p_con = save_p.replace('.mp4', '_con_particulas.mp4')

    def _render_pass(has_particles, out_path, desc_suffix):
        nonlocal cset
        leg.set_visible(has_particles)
        writer, encoder_name = _create_writer(fps)
        pbar = tqdm(total=len(frames), desc=f"Vídeo: Comparativa Vorticidad {desc_suffix}", position=0, leave=True)
        success = False
        try:
            with writer.saving(fig, out_path, dpi=DPI_SAVE):
                for idx in range(len(frames)):
                    if idx % FRAME_SKIP != 0:
                        pbar.update(1)
                        continue
                    zeta, psi, day, _ = frames[idx]
                    if np.isnan(zeta).any() or np.isnan(psi).any():
                        pbar.update(FRAME_SKIP)
                        continue
                    if np.max(np.abs(zeta)) < 1e-18 or np.max(np.abs(psi)) < 1e-18:
                        pbar.update(FRAME_SKIP)
                        continue

                    im.set_array(zeta.ravel())
                    if cset is not None:
                        cset.remove()
                    cset = ax.contour(X, Y, psi, colors='#2d2d2d', linewidths=1.2,
                                     alpha=0.5, levels=np.linspace(-vmax_psi, vmax_psi, 15),
                                     linestyles='solid')

                    if has_particles:
                        particles = _load_particle_data(all_modelos, snapshot_files[idx])
                        for suf, sc in scatter_groups.items():
                            px, py = particles.get(suf, (np.empty(0), np.empty(0)))
                            if len(px) > 0:
                                sc.set_offsets(np.column_stack((px, py)))
                            else:
                                sc.set_offsets(np.empty((0, 2)))
                    else:
                        for sc in scatter_groups.values():
                            sc.set_offsets(np.empty((0, 2)))

                    title_artist.set_text(f"Vorticidad - Comparativa - Día {day:.1f}")
                    writer.grab_frame()
                    pbar.update(FRAME_SKIP)
            success = True
        except Exception as e:
            print(f"\n[ERROR] Vídeo comparativa vorticidad {desc_suffix} ({encoder_name}): {e}")
        finally:
            pbar.close()
        if success:
            print(f"\n[OK] Vídeo comparativa vorticidad {desc_suffix} ({encoder_name}) -> {out_path}")
        sys.stdout.flush()

    _render_pass(False, save_p_sin, '(sin partículas)')
    _render_pass(True, save_p_con, '(con partículas)')

    plt.close(fig)
    plt.cla()
    plt.clf()
    gc.collect()


def save_comparison_psi_animation(snapshot_files, t_f, all_modelos, save_p, vmax_psi, fps):
    """Vídeo comparativo: función de corriente + partículas de 3 resoluciones."""
    modelos_dir_base = list(all_modelos.values())[-1]
    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir_base, f))
        if first_metadata:
            break
    x, y, X, Y = _compute_grid(first_metadata)
    n = len(snapshot_files)

    frames = []
    for i in range(n):
        try:
            data = np.load(os.path.join(modelos_dir_base, snapshot_files[i]))
            frames.append((data['psi'], t_f[i]))
            del data
            gc.collect()
        except Exception:
            pass

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI_SAVE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=CBAR_TICK_SIZE)
    ax.set_aspect('equal')

    vmax_p = vmax_psi
    im = ax.pcolormesh(X, Y, frames[0][0], cmap='coolwarm',
                      vmin=-vmax_p, vmax=vmax_p, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\psi$ [m$^2$/s]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(CBAR_TICK_SIZE)
    cset = ax.contour(X, Y, frames[0][0], colors='k', linewidths=0.9,
                     alpha=0.45, levels=15)

    scatter_groups = {}
    for n in all_modelos:
        color = COMPARISON_COLORS.get(n, '#B50000')
        sc = ax.scatter([], [], c=color, s=4, edgecolors='none',
                        alpha=0.8, zorder=5, label=COMPARISON_LABELS.get(n, f'N={n}'))
        scatter_groups[n] = sc

    leg = ax.legend(fontsize=CBAR_TICK_SIZE, loc='upper right', markerscale=3)

    title_artist = ax.set_title(r"Día 0.00", fontsize=VIDEO_TITLE_SIZE, pad=8, fontweight='bold')
    fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(CBAR_TICK_SIZE - 3, 9), color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    save_p_sin = save_p.replace('.mp4', '_sin_particulas.mp4')
    save_p_con = save_p.replace('.mp4', '_con_particulas.mp4')

    def _render_pass(has_particles, out_path, desc_suffix):
        nonlocal cset
        leg.set_visible(has_particles)
        writer, encoder_name = _create_writer(fps)
        pbar = tqdm(total=len(frames), desc=f"Vídeo: Comparativa Función Corriente {desc_suffix}", position=0, leave=True)
        success = False
        try:
            with writer.saving(fig, out_path, dpi=DPI_SAVE):
                for idx in range(len(frames)):
                    if idx % FRAME_SKIP != 0:
                        pbar.update(1)
                        continue
                    psi, day = frames[idx]
                    if np.isnan(psi).any():
                        pbar.update(FRAME_SKIP)
                        continue
                    if np.max(np.abs(psi)) < 1e-18:
                        pbar.update(FRAME_SKIP)
                        continue

                    im.set_array(psi.ravel())
                    if cset is not None:
                        cset.remove()
                    cset = ax.contour(X, Y, psi, colors='k', linewidths=0.9,
                                     alpha=0.45, levels=np.linspace(-vmax_p, vmax_p, 15),
                                     linestyles='solid')

                    if has_particles:
                        particles = _load_particle_data(all_modelos, snapshot_files[idx])
                        for suf, sc in scatter_groups.items():
                            px, py = particles.get(suf, (np.empty(0), np.empty(0)))
                            if len(px) > 0:
                                sc.set_offsets(np.column_stack((px, py)))
                            else:
                                sc.set_offsets(np.empty((0, 2)))
                    else:
                        for sc in scatter_groups.values():
                            sc.set_offsets(np.empty((0, 2)))

                    title_artist.set_text(f"Función de corriente $\\psi$ - Comparativa - Día {day:.1f}")
                    writer.grab_frame()
                    pbar.update(FRAME_SKIP)
            success = True
        except Exception as e:
            print(f"\n[ERROR] Vídeo comparativa psi {desc_suffix} ({encoder_name}): {e}")
        finally:
            pbar.close()
        if success:
            print(f"\n[OK] Vídeo comparativa psi {desc_suffix} ({encoder_name}) -> {out_path}")
        sys.stdout.flush()

    _render_pass(False, save_p_sin, '(sin partículas)')
    _render_pass(True, save_p_con, '(con partículas)')

    plt.close(fig)
    plt.cla()
    plt.clf()
    gc.collect()


def _load_trajectories_from_nc_all(nlist):
    """Precarga TODAS las trayectorias desde NetCDF para todas las resoluciones.

    Retorna dict: {n: (x_arr, y_arr)} con arrays shape (n_frames, n_particles),
    o {n: None} si no se encuentra el archivo NetCDF.
    """
    from netCDF4 import Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trajectories = {}
    for n in nlist:
        nc_path = os.path.join(base_dir, "output", PARTICLE_NC_DIR, PARTICLE_NC_PATTERN.format(n))
        if os.path.exists(nc_path):
            try:
                with Dataset(nc_path, 'r') as ds:
                    x = np.asarray(ds.variables['x'][:], dtype=np.float64)
                    y = np.asarray(ds.variables['y'][:], dtype=np.float64)
                trajectories[n] = (x, y)
                print(f"    NetCDF: {n} partículas, {x.shape[0]} frames cargados")
            except Exception as e:
                print(f"    AVISO: Error al leer {nc_path}: {e}")
                trajectories[n] = None
        else:
            trajectories[n] = None
    return trajectories


def save_comparison_velocity_animation(snapshot_files, t_f, all_modelos, save_p, vmax_vel, fps, nlist):
    """Vídeo comparativo: campo de velocidad + partículas de 3 resoluciones desde NetCDF."""
    modelos_dir_base = list(all_modelos.values())[-1]
    first_metadata = None
    for f in snapshot_files:
        first_metadata = _load_grid_metadata(os.path.join(modelos_dir_base, f))
        if first_metadata:
            break
    x_km, y_km, X, Y = _compute_grid(first_metadata)
    metadata = first_metadata or {}
    dx = metadata.get('dx', 6.0e6 / 512)
    dy = metadata.get('dy', 3.0e6 / 256)
    n = len(snapshot_files)

    # Precargar campos de velocidad desde snapshots
    vel_frames = []
    for i in range(n):
        try:
            data = np.load(os.path.join(modelos_dir_base, snapshot_files[i]))
            vel = _compute_velocity_field(data['psi'], dx, dy)
            vel_frames.append(vel)
            del data
            gc.collect()
        except Exception:
            pass

    if not vel_frames:
        print("  ERROR: No se pudieron cargar campos de velocidad.")
        return

    # Precargar trayectorias desde NetCDF (con fallback a snapshots)
    nc_traj = _load_trajectories_from_nc_all(nlist)
    use_nc = any(v is not None for v in nc_traj.values())

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI_SAVE)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Distancia [km]', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=CBAR_TICK_SIZE)
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 3000)
    ax.set_aspect('equal')

    vmax_v = vmax_vel
    im = ax.pcolormesh(X, Y, vel_frames[0], cmap='YlGnBu',
                      vmin=0.0, vmax=vmax_v, shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Velocidad [m/s]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(CBAR_TICK_SIZE)

    # Tres grupos de partículas con colores corporativos
    scatter_groups = {}
    for n_val in nlist:
        color = COMPARISON_COLORS.get(n_val, '#B50000')
        sc = ax.scatter([], [], c=color, s=4, edgecolors='none',
                        alpha=0.8, zorder=5, label=COMPARISON_LABELS.get(n_val, f'N={n_val}'))
        scatter_groups[n_val] = sc

    leg = ax.legend(fontsize=CBAR_TICK_SIZE, loc='upper right', markerscale=3)

    title_artist = ax.set_title(r"Día 0.00", fontsize=VIDEO_TITLE_SIZE, pad=8, fontweight='bold')
    fig.text(0.12, 0.02, AUTHOR_TEXT, fontsize=max(CBAR_TICK_SIZE - 3, 9), color='gray',
             ha='left', alpha=0.7, transform=fig.transFigure)

    save_p_sin = save_p.replace('.mp4', '_sin_particulas.mp4')
    save_p_con = save_p.replace('.mp4', '_con_particulas.mp4')

    def _render_pass(has_particles, out_path, desc_suffix):
        leg.set_visible(has_particles)
        writer, encoder_name = _create_writer(fps)
        pbar = tqdm(total=len(vel_frames), desc=f"Vídeo: Comparativa Velocidad {desc_suffix}", position=0, leave=True)
        success = False
        try:
            with writer.saving(fig, out_path, dpi=DPI_SAVE):
                for idx in range(len(vel_frames)):
                    if idx % FRAME_SKIP != 0:
                        pbar.update(1)
                        continue
                    vel = vel_frames[idx]
                    day = t_f[idx] if idx < len(t_f) else idx

                    if np.isnan(vel).any() or np.max(vel) < 1e-18:
                        pbar.update(FRAME_SKIP)
                        continue

                    im.set_array(vel.ravel())

                    if has_particles:
                        if use_nc:
                            for n_val, sc in scatter_groups.items():
                                traj = nc_traj.get(n_val)
                                if traj is not None and idx < traj[0].shape[0]:
                                    px = traj[0][idx] / 1e3
                                    py = traj[1][idx] / 1e3
                                    sc.set_offsets(np.column_stack((px, py)))
                                else:
                                    sc.set_offsets(np.empty((0, 2)))
                        else:
                            particles = _load_particle_data(all_modelos, snapshot_files[idx])
                            for suf, sc in scatter_groups.items():
                                px, py = particles.get(suf, (np.empty(0), np.empty(0)))
                                if len(px) > 0:
                                    sc.set_offsets(np.column_stack((px, py)))
                                else:
                                    sc.set_offsets(np.empty((0, 2)))
                    else:
                        for sc in scatter_groups.values():
                            sc.set_offsets(np.empty((0, 2)))

                    title_artist.set_text(f"Velocidad - Comparativa - Día {day:.1f}")
                    writer.grab_frame()
                    pbar.update(FRAME_SKIP)
            success = True
        except Exception as e:
            print(f"\n[ERROR] Vídeo comparativa velocidad {desc_suffix} ({encoder_name}): {e}")
        finally:
            pbar.close()
        if success:
            print(f"\n[OK] Vídeo comparativa velocidad {desc_suffix} ({encoder_name}) -> {out_path}")
        sys.stdout.flush()

    _render_pass(False, save_p_sin, '(sin partículas)')
    _render_pass(True, save_p_con, '(con partículas)')

    plt.close(fig)
    plt.cla()
    plt.clf()
    gc.collect()


def plot_comparison_dispersion_metrics(output_dir, metricas_dir, nlist):
    """Gráficas comparativas de dispersión — lee trayectorias y computa métricas dinámicamente."""
    os.makedirs(metricas_dir, exist_ok=True)

    data_series = {}
    for n in nlist:
        t_since, area_km2, rms_disp, centroids = _load_particle_dispersion(output_dir, n)
        if t_since is None:
            print(f"  AVISO: No se pudieron cargar datos de dispersión para n={n}")
            continue
        data_series[n] = {
            't_since': t_since,
            'area_km2': area_km2,
            'rms_disp': rms_disp / 1e3,
            'centroids': centroids,
            'label': COMPARISON_LABELS.get(n, f'N={n}'),
            'color': COMPARISON_COLORS.get(n, '#333333'),
        }

    if not data_series:
        print("  No hay datos de dispersión para las resoluciones encontradas.")
        return

    paper_fonts = {
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.titlesize': 16,
    }

    with plt.rc_context(paper_fonts):
        # Área de búsqueda
        fig1, ax1 = plt.subplots(figsize=(11.5, 5.8))
        for n in sorted(data_series):
            s = data_series[n]
            ax1.plot(s['t_since'], s['area_km2'], color=s['color'], lw=2.0,
                     marker='o', ms=2, label=s['label'])
            ax1.fill_between(s['t_since'], s['area_km2'], alpha=0.08, color=s['color'])
        ax1.set_xlabel('Tiempo desde el accidente [días]')
        ax1.set_ylabel('Área de búsqueda [km²]')
        ax1.set_title('Evolución del área de búsqueda - Comparativa', fontweight='bold', pad=12)
        ax1.ticklabel_format(style='plain', axis='y')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                  ha='left', alpha=0.7, transform=fig1.transFigure)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fp = os.path.join(metricas_dir, "search_area_comparison.svg")
        plt.savefig(fp, format='svg', bbox_inches='tight')
        print(f"  Guardado: search_area_comparison.svg")
        plt.close(fig1)

        # RMS Log-Log
        fig2, ax2 = plt.subplots(figsize=(11.5, 5.8))
        for n in sorted(data_series):
            s = data_series[n]
            ax2.loglog(s['t_since'] + 1, s['rms_disp'], color=s['color'], lw=2.0,
                      marker='o', ms=2, label=s['label'])
        ax2.set_xlabel('Tiempo desde el accidente [días]')
        ax2.set_ylabel('Radio RMS [km]')
        ax2.set_title('Dispersión RMS (escala log-log) - Comparativa', fontweight='bold', pad=12)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend()
        fig2.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                  ha='left', alpha=0.7, transform=fig2.transFigure)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fp = os.path.join(metricas_dir, "rms_dispersion_comparison.svg")
        plt.savefig(fp, format='svg', bbox_inches='tight')
        print(f"  Guardado: rms_dispersion_comparison.svg")
        plt.close(fig2)

        # Trayectoria del centroide
        fig3, ax3 = plt.subplots(figsize=(11.5, 5.8))
        from matplotlib.lines import Line2D
        for n in sorted(data_series):
            s = data_series[n]
            cx_km = s['centroids'][:, 0] / 1e3
            cy_km = s['centroids'][:, 1] / 1e3
            ax3.plot(cx_km, cy_km, color=s['color'], lw=2.0, marker='', label=s['label'])
            ax3.scatter(cx_km[0], cy_km[0], c=s['color'], s=140 if n == 10 else (110 if n == 100 else 80),
                       marker='o', edgecolors='black', linewidths=0.8, alpha=0.7, zorder=5)
            ax3.scatter(cx_km[-1], cy_km[-1], c=s['color'], s=140 if n == 10 else (110 if n == 100 else 80),
                       marker='s', edgecolors='black', linewidths=0.8, alpha=0.7, zorder=5)
        proxy_lines = [Line2D([], [], color=COMPARISON_COLORS[n], lw=2, label=COMPARISON_LABELS[n])
                       for n in sorted(data_series)]
        proxy_start = Line2D([], [], color='gray', marker='o', linestyle='None',
                             markersize=8, markeredgecolor='black', label='Inicio')
        proxy_end = Line2D([], [], color='gray', marker='s', linestyle='None',
                           markersize=8, markeredgecolor='black', label='Final')
        ax3.legend(handles=proxy_lines + [proxy_start, proxy_end], fontsize=10)
        ax3.set_xlabel('X [km]')
        ax3.set_ylabel('Y [km]')
        ax3.set_title('Trayectoria del centroide - Comparativa', fontweight='bold', pad=12)
        ax3.set_xlim(0, 6000)
        ax3.set_ylim(0, 3000)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        fig3.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                  ha='left', alpha=0.7, transform=fig3.transFigure)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fp = os.path.join(metricas_dir, "centroid_trajectory_comparison.svg")
        plt.savefig(fp, format='svg', bbox_inches='tight')
        print(f"  Guardado: centroid_trajectory_comparison.svg")
        plt.close(fig3)


def plot_comparison_energy_history(output_dir, metricas_dir):
    """Gráfica única de energía y enstrofía desde simulation_summary.npz (curva única)."""
    os.makedirs(metricas_dir, exist_ok=True)

    # Buscar el archivo simulation_summary.npz global
    summary_path = os.path.join(output_dir, "simulation_summary.npz")
    if not os.path.exists(summary_path):
        # Fallback: simulation_summary_res_{n}.npz (usar el de más partículas)
        alt_path = os.path.join(output_dir, "simulation_summary_res_1000.npz")
        if not os.path.exists(alt_path):
            alt_path = os.path.join(output_dir, "simulation_summary_res_100.npz")
        if not os.path.exists(alt_path):
            alt_path = os.path.join(output_dir, "simulation_summary_res_10.npz")
        if not os.path.exists(alt_path):
            print("  AVISO: No se encontró ningún archivo simulation_summary(.npz|_res_*.npz)")
            return
        summary_path = alt_path

    data = np.load(summary_path)
    t_days = data['energy_days']
    energy_vals = data['energy']
    enstrophy_vals = data['enstrophy']
    data.close()

    paper_fonts = {
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
        'figure.titlesize': 16,
    }
    with plt.rc_context(paper_fonts):
        fig, ax1 = plt.subplots(figsize=(11.5, 5.8))

        color_e = '#2E86AB'
        color_z = '#E94F37'

        l1 = ax1.plot(t_days, energy_vals, color=color_e, lw=2.0, label='Energía')
        ax1.set_xlabel('Tiempo [días]')
        ax1.set_ylabel('Energía cinética total [J/kg]', color=color_e)
        ax1.tick_params(axis='y', labelcolor=color_e)
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        l2 = ax2.plot(t_days, enstrophy_vals, color=color_z, lw=2.0, label='Enstrofía')
        ax2.set_ylabel('Enstrofía total [s⁻²]', color=color_z)
        ax2.tick_params(axis='y', labelcolor=color_z)
        ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))

        # Asíntotas del estado estacionario (días 500–2000)
        mask = (t_days >= 500) & (t_days <= 2000)
        if np.count_nonzero(mask) >= 2:
            energy_mean = np.mean(energy_vals[mask])
            enstrophy_mean = np.mean(enstrophy_vals[mask])
            ax1.axhline(energy_mean, color=color_e, linestyle='--', linewidth=1.6, alpha=0.65, label='Media energía 500–2000 días')
            ax2.axhline(enstrophy_mean, color=color_z, linestyle='--', linewidth=1.6, alpha=0.65, label='Media enstrofía 500–2000 días')
            from matplotlib.lines import Line2D
            h_energy_mean = Line2D([], [], color=color_e, ls='--', lw=1.6, alpha=0.65, label='Media energía 500–2000 días')
            h_enstrophy_mean = Line2D([], [], color=color_z, ls='--', lw=1.6, alpha=0.65, label='Media enstrofía 500–2000 días')
            handles = [l1[0], h_energy_mean, l2[0], h_enstrophy_mean]
        else:
            print("[WARN] No hay suficientes datos entre 500–2000 días; se omiten líneas de tendencia.")
            handles = [l1[0], l2[0]]
        ax1.legend(handles=handles, loc='upper right')

        ax1.set_title('Evolución de energía y enstrofía del modelo', fontweight='bold', pad=15)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        fig.text(0.12, 0.01, AUTHOR_TEXT, fontsize=9, color='gray',
                 ha='left', alpha=0.7, transform=fig.transFigure)

        save_path = os.path.join(metricas_dir, "energy_evolution_comparison.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close('all')
        gc.collect()
        print(f"  Guardado: energy_evolution_comparison.svg")


if __name__ == "__main__":
    post_process()