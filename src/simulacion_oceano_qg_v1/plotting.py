import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import animation
import numpy as np
import shutil
import importlib
from importlib.util import find_spec
from grid import Grid

def set_plot_style(profile: str = 'paper', use_seaborn: bool = False):
    """Configura estilos visuales predefinidos para memoria o presentación.
    
    profile:
        - 'paper': estilo sobrio para memoria/documento.
        - 'presentation': estilo más contrastado y legible en proyector.
    """
    if use_seaborn:
        if find_spec('seaborn') is not None:
            sns = importlib.import_module('seaborn')
            sns.set_theme(style='whitegrid', context='talk' if profile == 'presentation' else 'paper')
    
    common = {
        'figure.autolayout': True,
        'savefig.bbox': 'tight',
        'savefig.format': 'svg',
        'axes.grid': True,
        'grid.alpha': 0.18,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.2,
        'image.interpolation': 'bicubic',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'lines.antialiased': True,
        'patch.antialiased': True
    }
    
    paper = {
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'axes.titleweight': 'normal',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'lines.markersize': 5,
        'savefig.dpi': 200,
        'axes.prop_cycle': cycler(color=['#1f77b4', '#FF6B35', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    }
    
    presentation = {
        'font.family': 'DejaVu Sans',
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titleweight': 'semibold',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#cccccc',
        'lines.linewidth': 3.0,
        'lines.markersize': 8,
        'savefig.dpi': 240,
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#fafbfc',
        'axes.edgecolor': '#b0b0b0',
        'axes.prop_cycle': cycler(color=['#0A84FF', '#FF6B35', '#00C896', '#FF3B30', '#9D5FFF', '#FF9500'])
    }
    
    if profile not in ('paper', 'presentation'):
        raise ValueError("profile debe ser 'paper' o 'presentation'")
    
    style = common.copy()
    style.update(presentation if profile == 'presentation' else paper)
    plt.rcParams.update(style)

def _configure_ffmpeg():
    """Configura matplotlib para usar el binario de ffmpeg incluido en imageio-ffmpeg."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        mpl.rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()
        return
    except ImportError:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            return
    raise RuntimeError(
        "No se encontró ffmpeg. Ejecuta el proyecto con el venv de TFG o instala imageio-ffmpeg/ffmpeg."
    )

def _render_fields(ax, grid: Grid, zeta: np.ndarray, psi: np.ndarray, title: str):
    """Dibuja en un eje la vorticidad y los contornos de psi."""
    vmax = np.max(np.abs(zeta)) if np.any(zeta) else 1.0
    im = ax.pcolormesh(grid.X / 1e3, grid.Y / 1e3, zeta,
                       cmap='twilight_shifted', vmin=-vmax, vmax=vmax, shading='auto')
    ax.contour(grid.X / 1e3, grid.Y / 1e3, psi, colors='#2d2d2d', linewidths=1.0, alpha=0.5, levels=12)
    ax.set_title(title, pad=12)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_aspect('equal')
    return im

def _render_complete(ax, grid: Grid, zeta: np.ndarray, psi: np.ndarray,
                     px: np.ndarray, py: np.ndarray, t_days: float, title: str):
    """Dibuja en un eje los campos y la nube de partículas."""
    vmax = np.max(np.abs(zeta)) if np.any(zeta) else 1.0
    im = ax.pcolormesh(grid.X / 1e3, grid.Y / 1e3, zeta,
                       cmap='twilight_shifted', vmin=-vmax, vmax=vmax, shading='auto')
    ax.contour(grid.X / 1e3, grid.Y / 1e3, psi, colors='#333333', linewidths=0.8, alpha=0.4, levels=12)
    ax.scatter(px / 1e3, py / 1e3, c='#FF6B35', s=6, edgecolors='#D62828', linewidths=0.3, alpha=0.8, label='Restos flotantes')
    ax.text(0.02, 0.95, f'Día: {t_days:.2f}', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    ax.set_title(title, pad=12)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_aspect('equal')
    return im

def plot_fields(grid: Grid, zeta: np.ndarray, psi: np.ndarray, title: str = "Campos Eulerianos", save_path: str = None):
    """Vorticidad (twilight mejorado) y contornos de Función de Corriente."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = _render_fields(ax, grid, zeta, psi, title)
    cbar = plt.colorbar(im, ax=ax, label=r'Vorticidad $\zeta$ [s$^{-1}$]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    if save_path:
        fmt = 'png' if save_path.lower().endswith('.png') else 'svg'
        plt.savefig(save_path, format=fmt)
    return fig, ax

def plot_streamfunction(grid: Grid, psi: np.ndarray, title: str = "Función de Corriente", save_path: str = None):
    """Mapa dedicado de función de corriente psi con contornos rellenos."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    vmax = np.max(np.abs(psi)) if np.any(psi) else 1.0
    im = ax.pcolormesh(grid.X / 1e3, grid.Y / 1e3, psi,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    ax.contour(grid.X / 1e3, grid.Y / 1e3, psi, colors='#2d2d2d', linewidths=0.9, alpha=0.5, levels=14)
    cbar = plt.colorbar(im, ax=ax, label=r'$\psi$ [m$^2$/s]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(title, pad=12)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_aspect('equal')
    if save_path:
        fmt = 'png' if save_path.lower().endswith('.png') else 'svg'
        plt.savefig(save_path, format=fmt)
    return fig, ax

def plot_velocity_field(grid: Grid, u: np.ndarray, v: np.ndarray, save_path: str = None):
    """Mapa de velocidad usando streamplot coloreado por módulo."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    speed = np.sqrt(u**2 + v**2)
    strm = ax.streamplot(grid.x/1e3, grid.y/1e3, u.T, v.T, color=speed.T, 
                         cmap='plasma', linewidth=1.3, density=1.3, arrowsize=1.5)
    cbar = plt.colorbar(strm.lines, ax=ax, label='Velocidad [m/s]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("Campo de Velocidades Geostróficas", pad=12)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_aspect('equal')
    if save_path:
        fmt = 'png' if save_path.lower().endswith('.png') else 'svg'
        plt.savefig(save_path, format=fmt)
    return fig, ax

def plot_snapshot_complete(grid: Grid, zeta: np.ndarray, psi: np.ndarray, px: np.ndarray, py: np.ndarray, t_days: float, save_path: str = None):
    """Composición final para la memoria: Campos + Nube de Partículas."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = _render_complete(ax, grid, zeta, psi, px, py, t_days, "Transporte Lagrangiano en el flujo QG")
    cbar = plt.colorbar(im, ax=ax, label=r'$\zeta$ [s$^{-1}$]', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    if save_path:
        fmt = 'png' if save_path.lower().endswith('.png') else 'svg'
        plt.savefig(save_path, format=fmt)
    return fig, ax

def save_fields_animation(grid: Grid, zeta_frames: list, psi_frames: list, t_days: list,
                          save_path: str, fps: int = 6, dpi: int = 150):
    """Guarda una animación MP4 de vorticidad y función de corriente."""
    _configure_ffmpeg()
    # Full HD: 1920x1080 = 12.8in x 7.2in @ 150 dpi
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    with writer.saving(fig, save_path, dpi=dpi):
        for zeta, psi, day in zip(zeta_frames, psi_frames, t_days):
            fig.clf()
            ax = fig.add_subplot(111)
            im = _render_fields(ax, grid, zeta, psi, f"Campos Eulerianos - Día {day:.2f}")
            fig.colorbar(im, ax=ax, label=r'Vorticidad $\zeta$ [s$^{-1}$]')
            writer.grab_frame()
    plt.close(fig)

def save_streamfunction_animation(grid: Grid, psi_frames: list, t_days: list,
                                  save_path: str, fps: int = 6, dpi: int = 150):
    """Guarda una animación MP4 dedicada a la función de corriente psi."""
    _configure_ffmpeg()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    with writer.saving(fig, save_path, dpi=dpi):
        for psi, day in zip(psi_frames, t_days):
            fig.clf()
            ax = fig.add_subplot(111)
            vmax = np.max(np.abs(psi)) if np.any(psi) else 1.0
            im = ax.pcolormesh(grid.X / 1e3, grid.Y / 1e3, psi,
                               cmap='coolwarm', vmin=-vmax, vmax=vmax, shading='auto')
            ax.contour(grid.X / 1e3, grid.Y / 1e3, psi, colors='k', linewidths=0.6, alpha=0.45)
            ax.set_title(f"Función de Corriente - Día {day:.2f}")
            ax.set_xlabel('x [km]')
            ax.set_ylabel('y [km]')
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax, label=r'$\psi$ [m$^2$/s]')
            writer.grab_frame()
    plt.close(fig)

def save_complete_animation(grid: Grid, zeta_frames: list, psi_frames: list,
                            px_frames: list, py_frames: list, t_days: list,
                            save_path: str, fps: int = 6, dpi: int = 150):
    """Guarda una animación MP4 con campos y partículas."""
    _configure_ffmpeg()
    # Full HD: 1920x1080 = 12.8in x 7.2in @ 150 dpi
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    with writer.saving(fig, save_path, dpi=dpi):
        for zeta, psi, px, py, day in zip(zeta_frames, psi_frames, px_frames, py_frames, t_days):
            fig.clf()
            ax = fig.add_subplot(111)
            im = _render_complete(ax, grid, zeta, psi, px, py, day, "Transporte Lagrangiano en el flujo QG")
            fig.colorbar(im, ax=ax, label=r'$\zeta$ [s$^{-1}$]')
            writer.grab_frame()
    plt.close(fig)

def save_velocity_animation(grid: Grid, u_frames: list, v_frames: list, t_days: list,
                            save_path: str, fps: int = 6, dpi: int = 150):
    """Guarda una animación MP4 del campo de velocidades (streamplot coloreado)."""
    _configure_ffmpeg()
    # Full HD: 1920x1080 = 12.8in x 7.2in @ 150 dpi
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    with writer.saving(fig, save_path, dpi=dpi):
        for u, v, day in zip(u_frames, v_frames, t_days):
            fig.clf()
            ax = fig.add_subplot(111)
            speed = np.sqrt(u**2 + v**2)
            strm = ax.streamplot(grid.x/1e3, grid.y/1e3, u.T, v.T, color=speed.T,
                                cmap='plasma', linewidth=1.5, density=1.4, arrowsize=1.8)
            ax.set_title(f"Campo de Velocidades Geostróficas - Día {day:.2f}", fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('x [km]', fontsize=14)
            ax.set_ylabel('y [km]', fontsize=14)
            ax.set_aspect('equal')
            cbar = fig.colorbar(strm.lines, ax=ax, label='Velocidad [m/s]', pad=0.02)
            cbar.ax.tick_params(labelsize=11)
            writer.grab_frame()
    plt.close(fig)

def plot_dispersion_metrics(t_days: np.ndarray, msd_vals: np.ndarray, r_rms_vals: np.ndarray, save_path: str = None):
    """Gráfica de evolución temporal de MSD y Radio RMS para la memoria."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(t_days, msd_vals, color='#0A84FF', lw=3, marker='o', markersize=4, markeredgecolor='#0052B4', markeredgewidth=1, alpha=0.85)
    ax1.fill_between(t_days, msd_vals, alpha=0.15, color='#0A84FF')
    ax1.set_ylabel(r'MSD [m$^2$]', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_title("Evolución de Métricas de Dispersión", fontweight='bold', pad=12)
    ax2.plot(t_days, r_rms_vals, color='#FF6B35', lw=3, marker='s', markersize=4, markeredgecolor='#D62828', markeredgewidth=1, alpha=0.85)
    ax2.fill_between(t_days, r_rms_vals, alpha=0.15, color='#FF6B35')
    ax2.set_ylabel('Radio RMS [m]', fontweight='bold')
    ax2.set_xlabel('Tiempo [días]', fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    if save_path:
        if not save_path.lower().endswith('.svg'):
            save_path = save_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(save_path, format='svg')
    return fig

def plot_energy_timeseries(t_days: np.ndarray, energy_vals: np.ndarray, save_path: str = None):
    """Evolución temporal de la energía cinética integrada."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t_days, energy_vals, color='#00C896', lw=3, marker='D', markersize=5, markeredgecolor='#007F5F', markeredgewidth=1, alpha=0.85)
    ax.fill_between(t_days, energy_vals, alpha=0.15, color='#00C896')
    ax.set_title('Evolución de Energía Cinética', fontweight='bold', pad=12)
    ax.set_xlabel('Tiempo [días]', fontweight='bold')
    ax.set_ylabel(r'Energía cinética [m$4$/s$2$]', fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    if save_path:
        if not save_path.lower().endswith('.svg'):
            save_path = save_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(save_path, format='svg')
    return fig
